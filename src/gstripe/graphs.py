from __future__ import annotations

from typing import Dict, Tuple
from typing import Hashable

import igraph as ig
import numpy as np
import pandas as pd

from .ranges import RangeSeries, overlapping_clusters_grouped
from .utils import pandas_merge_threeway


class AnchorClustering(object):
    def __call__(self, anchors):
        cluster_ids = pd.Series(np.arange(len(anchors), dtype=int), index=anchors.index, copy=False)
        return anchors, cluster_ids


class ClusterOverlappingAnchors(AnchorClustering):
    def __init__(self, expand_by=0, expand_from_midpoint=False):
        super().__init__()
        self.expand_by = expand_by
        self.expand_from_midpoint = expand_from_midpoint

    def __call__(self, anchors_df):
        if self.expand_from_midpoint:
            anchors_df = anchors_df.sort_values(['chromosome', 'midpoint'])
            anchors = RangeSeries(anchors_df.midpoint)
        else:
            anchors_df = anchors_df.sort_values(['chromosome', 'start', 'end'])
            anchors = RangeSeries(anchors_df.start, anchors_df.end)
        anchors = anchors.expand(self.expand_by)
        cluster_idxs = overlapping_clusters_grouped(anchors, anchors_df.chromosome)
        return anchors_df, cluster_idxs


class StrandEdgesAddition(object):
    def __call__(self, nodes, edges):
        return nodes, edges


class AddAllStrandEdges(StrandEdgesAddition):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, nodes, edges):
        n = len(nodes) - 1
        strand_edge_length = nodes.midpoint.iloc[1:].to_numpy() - nodes.midpoint.iloc[:-1].to_numpy()
        strand_edges_cols = {
            'node_A': nodes.index[:-1],
            'node_B': nodes.index[1:],
            'length': strand_edge_length,
            'is_contact': np.full(n, False, dtype=bool),
            'is_strand': np.full(n, True, dtype=bool),
            'n_contacts': np.zeros(n, dtype=int)
        }
        all_cols = create_new_columns_with_defaults(strand_edges_cols, n)
        strand_edges = pd.DataFrame(all_cols)
        strand_edges = strand_edges.set_index(['node_A', 'node_B'], drop=True)
        edges = pd.concat([edges, strand_edges]).sort_index()
        return nodes, edges


class CreateGraph(object):
    def __init__(
            self,
            add_strand_edges: StrandEdgesAddition = None,
            anchor_clustering: AnchorClustering = ClusterOverlappingAnchors(),
            anchors_data_aggregation: Dict[Hashable, Hashable] = None,
            contacts_data_aggregation: Dict[Hashable, Hashable] = None,
            remove_loops: bool = False,
    ):
        self.add_strand_edges = add_strand_edges if add_strand_edges is not None else StrandEdgesAddition()
        self.anchor_clustering = anchor_clustering
        self.anchors_data_aggregation = anchors_data_aggregation
        self.contacts_data_aggregation = contacts_data_aggregation
        self.remove_loops = remove_loops

    def __call__(
            self,
            anchors: pd.DataFrame,
            contacts: pd.DataFrame
    ) -> Tuple[ig.Graph, pd.DataFrame, pd.DataFrame]:
        nodes, anchors, contacts = self.create_nodes(anchors, contacts, self.anchors_data_aggregation)
        nodes = self.add_extra_nodes(nodes)
        edges, anchors, contacts = self.create_edges(anchors, contacts, self.contacts_data_aggregation)
        edges = self.add_extra_edges(edges)
        nodes, edges = self.add_strand_edges(nodes, edges)
        graph = ig.Graph.DataFrame(
            edges.reset_index(),
            vertices=nodes.reset_index(),
            directed=False,
            use_vids=True
        )
        return graph, anchors, contacts

    def add_extra_nodes(self, nodes):
        return nodes

    def add_extra_edges(self, edges):
        return edges

    def create_nodes(
            self,
            anchors: pd.DataFrame, contacts: pd.DataFrame,
            anchors_data_aggregation
    ):
        anchors, cluster_ids = self.anchor_clustering(anchors)
        cluster_ids = pd.Series(cluster_ids, name='node_id')
        anchors = pd.concat([cluster_ids, anchors.reset_index()], axis=1)
        anchors = anchors.sort_values(['node_id', 'anchor_id'], ignore_index=True)
        data_agg = self._make_pandas_aggregation(
            anchors_data_aggregation, self.default_anchors_data_aggregation()
        )
        nodes = anchors.groupby('node_id', observed=True).agg(**data_agg)
        nodes['midpoint'] = RangeSeries(nodes.start, nodes.end).center
        return nodes, anchors, contacts

    def create_edges(
            self,
            anchors: pd.DataFrame, contacts: pd.DataFrame,
            contacts_data_aggregation
    ):
        _anchors_ri = anchors  # .reset_index()  # TODO: index????
        merged_contacts = pandas_merge_threeway(
            _anchors_ri, contacts.reset_index(), _anchors_ri,
            mid_to_left='anchor_id_A', mid_to_right='anchor_id_B',
            left_on='anchor_id', right_on='anchor_id',
            suffixes=('_A', '', '_B')
        )
        data_agg = self._make_pandas_aggregation(
            contacts_data_aggregation, self.default_contacts_data_aggregation()
        )
        id_cols = ['node_id_A', 'node_id_B']
        edges = merged_contacts.groupby(id_cols, observed=True).agg(**data_agg).reset_index(id_cols)
        edges['is_contact'] = True
        edges['is_strand'] = False
        edges['weight'] = edges.total_petcount.astype(float)
        if self.remove_loops:
            edges = edges.loc[edges.node_id_A != edges.node_id_B, :]
        edges = edges.set_index(id_cols)
        return edges, anchors, contacts

    def default_anchors_data_aggregation(self):
        return {
            'chromosome': pd.NamedAgg(column='chromosome', aggfunc='first'),
            'start': pd.NamedAgg(column='start', aggfunc='min'),
            'end': pd.NamedAgg(column='end', aggfunc='max'),
            'max_length': pd.NamedAgg(column='length', aggfunc='max'),
            'n_anchors': pd.NamedAgg(column='anchor_id', aggfunc='size')
        }

    def default_contacts_data_aggregation(self):
        return {
            'n_contacts': pd.NamedAgg(column='contact_id', aggfunc='size'),
            'total_petcount': pd.NamedAgg(column='petcount', aggfunc='sum'),
            'min_petcount': pd.NamedAgg(column='petcount', aggfunc='min'),
            'max_petcount': pd.NamedAgg(column='petcount', aggfunc='max'),
            'min_length': pd.NamedAgg(column='length', aggfunc='min'),
            'max_length': pd.NamedAgg(column='length', aggfunc='max')
        }

    @staticmethod
    def _make_pandas_aggregation(data_agg, default_data_agg=None):
        merged = dict(default_data_agg if default_data_agg is not None else [])
        if data_agg is not None:
            merged.update(data_agg)
        result = {
            col: agg if isinstance(agg, pd.NamedAgg) else pd.NamedAgg(column=col, aggfunc=agg)
            for col, agg in merged.items()
        }
        return result


def create_new_columns_with_defaults(cols, n, default_int=0, default_float=np.nan, default_obj=None, **column_dtypes):
    new_cols = dict(cols)
    for col, dt in column_dtypes.items():
        if col in cols:
            continue
        if pd.api.types.is_integer_dtype(dt):
            vals = np.full(n, default_int, dtype=dt)
        elif pd.api.types.is_float_dtype(dt):
            vals = np.full(n, default_float, dtype=dt)
        else:
            vals = np.full(n, default_obj, dtype=dt)
        new_cols[col] = vals
    return new_cols
