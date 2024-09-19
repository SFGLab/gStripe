#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import itertools
import operator
import os
import re
import sys
import logging
from dataclasses import dataclass
from enum import Enum
from operator import itemgetter
from typing import List, Iterable, Tuple, Dict, Any, Union, Optional

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import default_rng

from .ranges import RangeSeries
from .graphs import CreateGraph, ClusterOverlappingAnchors
from .task import TaskRunner, Timer, Task, MessagePrinter
from .utils import add_argparse_args, from_argparse_args, LOG_LEVELS
from .genomic_tools import read_raw_contacts_from_bedpe, normalize_contacts

rng = default_rng()


def setup_logger(logger, log_file=None, file_level=logging.DEBUG, console_level=logging.INFO):
    # log_levels = logging.getLevelNamesMapping()  # needs Python 3.11...
    if isinstance(file_level, str):
        file_level = LOG_LEVELS[file_level.upper()]
    if isinstance(console_level, str):
        console_level = LOG_LEVELS[console_level.upper()]

    log_format = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    level = console_level

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        level = min(level, file_level)

    logger.setLevel(level)    


class Attrs(Enum):
    """
    Helper class for getting attribute values from an igraph.Graph vertices or edges.
    """
    # VERTEX ATTRS
    CHROMOSOME = 'chromosome'
    ANCHOR_START = 'start'
    ANCHOR_END = 'end'
    NODE_ID = 'node_id'
    # EDGE ATTRS
    WEIGHT = 'total_petcount'
    LENGTH = 'max_length'  # TODO: remove
    LOOP_COUNT = 'n_contacts'
    EDGE_ID = 'edge_id'

    def get(self, entity):
        """
        Get the attribute value for a single vertex or edge.

        Args:
            entity: vertex or edge of an igraph.Graph

        Returns:
            The attribute value.
        """
        return entity[self.value]

    def get_all(self, entities_iterable):
        """
        Iterate over the attribute value for an iterable of vertices or edges.

        Args:
            entities_iterable: Iterable of vertics or edges of an igraph.Graph

        Returns:
            An iterator over the values of the attribute for the provided input iterable.
        """
        return (self.get(entity) for entity in entities_iterable)


def create_chromosome_graphs(bedpe_df: pd.DataFrame, anchor_expansion: int) -> Tuple[str, ig.Graph]:
    """
    Create graph representation of each chromosome in input interaction dataframe.

    Args:
        bedpe_df: Interaction dataframe, like the one obtained rom bedpe file.
        anchor_expansion: The distance by which each anchor should be expanded (on both sides) before anchor clustering.
    
    Returns:
        Yields tuples of the form (chromosome_name, graph)
    """
    graph_creator = CreateGraph(
        add_strand_edges=None,
        anchor_clustering=ClusterOverlappingAnchors(expand_by=anchor_expansion, expand_from_midpoint=False),
        remove_loops=True
    )
    for raw_label, chrom_df in bedpe_df.groupby('chromosome_A', observed=True):
        anchors, contacts = normalize_contacts(chrom_df)
        graph, *_ = graph_creator(anchors, contacts)
        yield raw_label, graph


def vertices_sorted_by_attribute(vertices, attr, reverse=False):
    """
    Get igraph vertices sorted by arttribute value.
    """
    if isinstance(attr, Attrs):  # TODO: move to graphs, not only vertices
        attr = attr.value
    return sorted(vertices, key=lambda v: v[attr], reverse=reverse)


class StripeDirection(Enum):
    """
    Represents the direction of the stripe: R or L.
    """
    right = ('R', True, operator.le, 'horizontal') # equal values should not happen, but will be put in 'R'
    left = ('L', False, operator.gt, 'vertical')

    def __init__(self, code, is_increasing, compare_op, alias):
        self.code = code
        self.is_increasing = is_increasing  # position_increases_with_length
        self.compare_op = compare_op  # anchor_to_leafs_relation
        self.alias = alias

    def __str__(self):
        return self.code

    def __eq__(self, other):
        if isinstance(other, StripeDirection):
            return self.code == other.code
        elif isinstance(other, str):
            return self.code == other
        else:
            return False

    def __hash__(self):
        return hash(self.code)

    def sort(self, collection, key=lambda x: x):
        lst = list(collection)
        lst.sort(key=key, reverse=not self.is_increasing)
        return lst

    def sort_vertices(self, collection):
        attr = Attrs.ANCHOR_START if self.is_increasing else Attrs.ANCHOR_END
        return self.sort(collection, key=lambda v: attr.get(v))


class Fan(object):
    def __init__(self, graph: ig.Graph, direction: StripeDirection, anchor: ig.Vertex, leafs: Iterable[ig.Vertex]):
        if graph is None or anchor is None:
            raise ValueError("Must provide graph and anchor")
        self.graph = graph        
        self.anchor = anchor
        leafs = direction.sort_vertices(leafs)
        if len(leafs) < 1:
            raise ValueError("Must provide at least one leaf")
        self.leafs = leafs
        self.edge_ids = [graph.get_eid(anchor, v) for v in self.leafs]
        self.edges = [graph.es[eid] for eid in self.edge_ids]
        self.chromosome = Attrs.CHROMOSOME.get(self.anchor)
        self.anchor_start = Attrs.ANCHOR_START.get(self.anchor)
        self.anchor_end = Attrs.ANCHOR_END.get(self.anchor)
        self.leafs_start = min(Attrs.ANCHOR_START.get_all(self.leafs))
        self.leafs_end = max(Attrs.ANCHOR_END.get_all(self.leafs))
        self.direction = direction
        pos_attr = Attrs.ANCHOR_START if direction.is_increasing else Attrs.ANCHOR_END
        self.positions = list(pos_attr.get_all(self.leafs))

    def __len__(self):
        return len(self.leafs)

    def __hash__(self):
        return hash((self.direction, Attrs.NODE_ID.get(self.anchor)))

    def __repr__(self):
        return f"<{self.direction}|{len(self)}|{self.anchor_start}-{self.anchor_end}|{self.leafs_start}-{self.leafs_end}>"


class LoopCluster(object):
    def __init__(self, leafs, edges, edge_ids, fans_by_leaf):
        self.leafs = list(leafs)
        self.edges = list(edges)
        self.edge_ids = list(edge_ids)
        self.unique_leafs = get_unique_vertices(self.leafs)
        self.fans_by_leaf = list(fans_by_leaf)
        self.fans = set(self.fans_by_leaf)
        fs = min(f.anchor_start for f in self.fans)
        fe = max(f.anchor_end for f in self.fans)
        ls = min(Attrs.ANCHOR_START.get_all(self.leafs))
        le = max(Attrs.ANCHOR_END.get_all(self.leafs))
        self.anchor_start = fs
        self.anchor_end = fe
        self.leafs_start = ls
        self.leafs_end = le
        if fs <= ls:
            self.left_start = fs
            self.left_end = fe
            self.right_start = ls
            self.right_end = le
        else:
            self.left_start = ls
            self.left_end = le
            self.right_start = fs
            self.right_end = fe
        self.length = self.leafs_end - self.leafs_start
        self.width = self.anchor_end - self.anchor_start

    def __len__(self):
        return len(self.edges)

    @classmethod
    def from_blade_cluster(cls, blade_cluster):
        vertices = []
        edges = []
        edge_ids = []
        fans_by_leaf = []
        for _, v, e, eid, fan in blade_cluster:
            vertices.append(v)
            edges.append(e)
            edge_ids.append(eid)
            fans_by_leaf.append(fan)
        return LoopCluster(vertices, edges, edge_ids, fans_by_leaf)


def get_unique_vertices(vertices, id_attr=Attrs.NODE_ID):
    ids = set()
    unique_verts = []
    for v in vertices:
        _id = id_attr.get(v)
        if _id not in ids:
            unique_verts.append(v)
            ids.add(_id)
    return unique_verts


class Stripe(object):
    def __init__(
        self,
        graph: ig.Graph,
        loop_clusters: Iterable[LoopCluster],
        stripe_id: int = -1
    ):
        assert graph is not None
        self._id = stripe_id
        self.graph = graph
        self.loop_clusters = list(loop_clusters)
        self.orthogonal_stripes = [(None, -1) for _ in range(len(self.loop_clusters))]
        assert len(self.loop_clusters) > 0
        self.fans = sorted(set(f for lc in self.loop_clusters for f in lc.fans), key=lambda f: f.anchor_start)
        assert len(self.fans) > 0
        self.direction = self.fans[0].direction
        assert all(f.direction == self.direction for f in self.fans)
        self.chromosome = self.fans[0].chromosome
        assert all(f.chromosome == self.chromosome for f in self.fans)

        # width (always increasing)
        self.anchor_start = min(f.anchor_start for f in self.fans)
        self.anchor_end = max(f.anchor_end for f in self.fans)
        self.position = (self.anchor_start + self.anchor_end) // 2

        # leaf segment extents (in order from anchor to end of stripe)
        l_min = min(lc.leafs_start for lc in self.loop_clusters)
        l_max = max(lc.leafs_end for lc in self.loop_clusters)
        if self.direction.is_increasing:  # R
            self.leafs_start = l_min
            self.leafs_end = l_max
        else:
            self.leafs_start = l_max
            self.leafs_end = l_min

        # length (always increasing)
        if self.direction.is_increasing:  # R
            self.start = self.anchor_start
            self.end = self.leafs_end
        else:  # L
            self.start = self.leafs_end
            self.end = self.anchor_end

        # features
        self.data_start = None
        self.data_end = None
        self.features = {}  # namespace for features to be assigned
        self.n_fans = len(self.fans)
        self.n_edges = sum(len(lc) for lc in self.loop_clusters)
        self.n_leafs = len(get_unique_vertices(v for cl in self.loop_clusters for v in cl.unique_leafs))

    @property
    def id(self):
        return self._id

    @property
    def length(self):
        return self.end - self.start

    @property
    def width(self):
        return self.anchor_end - self.anchor_start

    def __len__(self):
        return len(self.loop_clusters)

    def __repr__(self):
        return f"<{self.direction}-{self.id}|{self.n_fans}|{self.n_edges}|{self.anchor_start}-{self.anchor_end}|{self.start}-{self.end}>"

    def trim(self, end: int):
        assert end >= 0
        assert end <= len(self.loop_clusters)
        trimmed = Stripe(
            self.graph,
            self.loop_clusters[:end],
            stripe_id=self.id
        )
        for name, data in self.features.items():
            trimmed.features[name] = data[:end]
        return trimmed

    def merge(self, other: Stripe) -> Stripe:
        if other is None:
            return self
        if other.anchor_start < self.anchor_start:
            return other.merge(self)
        assert self.direction == other.direction
        sorted_lc = self.direction.sort(
            enumerate(self.loop_clusters + other.loop_clusters),
            key=lambda i_lc: i_lc[1].leafs_start
        )
        new_loop_clusters = [lc for _, lc in sorted_lc]
        new_stripe = Stripe(self.graph, new_loop_clusters, min(self.id, other.id))
        for name, data in self.features.items():
            concated = np.concatenate([data, other.features[name]])
            new_data = np.zeros(len(new_loop_clusters), dtype=data.dtype)
            for j, (i, _) in enumerate(sorted_lc):
                new_data[j] = concated[i]
            new_stripe.features[name] = new_data

        return new_stripe

def fans_in_graph(
        graph: ig.Graph,
        min_size=1,
        max_size=sys.maxsize - 1,
        position_attribute=Attrs.ANCHOR_START
) -> Tuple[List[Fan], List[Fan]]:
    assert min_size >= 1
    assert max_size >= min_size

    left_fans = []
    right_fans = []

    for anchor in vertices_sorted_by_attribute(graph.vs(), attr=position_attribute):
        left_leafs = []
        right_leafs = []
        is_left = StripeDirection.left.compare_op
        for v in anchor.neighbors():
            if is_left(position_attribute.get(anchor), position_attribute.get(v)):
                left_leafs.append(v)  # equal values should not happen, but will be put in 'R'
            else:
                right_leafs.append(v)
        if min_size <= len(left_leafs) <= max_size:
            left_fans.append(Fan(graph, StripeDirection.left, anchor, left_leafs))
        if min_size <= len(right_leafs) <= max_size:
            right_fans.append(Fan(graph, StripeDirection.right, anchor, right_leafs))

    return right_fans, left_fans


def cluster_sorted_objects_by_position(
        objects: List[Any],
        get_position=lambda o: o,
        max_span=0
) -> List[List[Any]]:
    cluster_list = []
    if len(objects) == 0:
        return cluster_list
    prev_start = objects[0]
    cluster = [prev_start]
    last_pos = get_position(prev_start)
    ascending = None
    for obj in objects[1:]:
        pos = get_position(obj)
        if pos != last_pos:
            asc = pos > last_pos
            if ascending is not None:
                assert ascending == asc, "Inconsistent ordering"
            else:
                ascending = asc
        if abs(pos - last_pos) <= max_span:
            cluster.append(obj)
        else:
            cluster_list.append(cluster)
            cluster = [obj]
            last_pos = pos
    cluster_list.append(cluster)
    return cluster_list


def get_sort_order(objects, key=lambda o: o):
    if len(objects) < 2:
        return 'Constant'
    is_ascending = None
    last_pos = key(objects[0])
    for obj in objects:
        pos = key(obj)
        if pos != last_pos:
            asc = pos > last_pos
            if is_ascending is not None and asc != is_ascending:
                return 'Unsorted'
            is_ascending = asc
            last_pos = pos
    if is_ascending is None:
        return 'Constant'
    return 'Ascending' if is_ascending else 'Descending'


def cluster_by_max_gap(
    objects: List[Any],
    max_gap,
    key=lambda o: o
) -> List[List[Any]]:
    cluster_list = []
    if len(objects) == 0:
        return cluster_list
    cluster = [objects[0]]
    last_pos = key(objects[0])
    for obj in objects[1:]:
        pos = key(obj)
        if abs(pos - last_pos) <= max_gap:
            cluster.append(obj)
        else:
            cluster_list.append(cluster)
            cluster = [obj]
            last_pos = pos
    cluster_list.append(cluster)
    return cluster_list


def get_chromosome_label_and_index(raw_label: str) -> Tuple[str, int]:
    m = re.match(r'(chr)?(\d+|X|Y|M)', raw_label)
    if m is None or len(m.group(2)) == 0:
        return raw_label, sys.maxsize - 1
    if len(m.group(1)) == 0:
        label = 'chr' + raw_label
    else:
        label = raw_label
    ch = m.group(2)
    symbols = {'X': 23, 'Y': 24, 'M': 25}
    if ch in symbols.keys():
        return label, symbols[ch]
    else:
        return label, int(ch)


@dataclass
class GraphStripeCaller(Task):
    fix_bin_start: bool = True
    skiprows: int = -1
    petcount_col: int = -1
    data_bin_size: int = 5000
    max_fan_cluster_gap: int = 15_000
    max_loop_cluster_gap: int = 15_000
    max_stripe_width: int = 40_000
    min_stripe_length: int = 20_000
    max_stripe_length: int = -1
    n_quantile_levels: int = 50
    stripe_gap_threshold: float = 0.0
    stripe_score_holder_exponent: float = 2.0
    describe_percentiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    merge_stripes_threshold: int = 0
    max_merged_stripe_width: int = 80_000

    # Features
    _f_length = None
    _f_loop_count = None
    _f_weight = None
    _f_gap_size = None
    _f_orientation_score = None
    _f_raw_gap_score = None
    _f_gap_score = None
    _f_stripe_score = None
    _f_worst_gap_score = None
    _f_l_score = None
    _f_w_score = None
    _f_cross_score = None
    
    def prepare_interactions(self, bedpe_df):
        petcount_col_idx = 6
        if 'petcount' not in bedpe_df.columns:
            bedpe_df.insert(petcount_col_idx, 'petcount', 1.0)
        if self.fix_bin_start:
            bedpe_df.loc[:, 'start_A'] += 1
            bedpe_df.loc[:, 'start_B'] += 1
        first_bin_coord = 1
        RangeSeries(bedpe_df, suffix='_A').bin(self.data_bin_size, first_bin_coord).assign_to(bedpe_df)
        RangeSeries(bedpe_df, suffix='_B').bin(self.data_bin_size, first_bin_coord).assign_to(bedpe_df)
        return bedpe_df

    def create_graphs(self, bedpe_df: pd.DataFrame) -> Dict[str, ig.Graph]:
        with self.timer('create_graphs') as t:
            graphs = [
                (*get_chromosome_label_and_index(raw_label), raw_label, graph)
                for raw_label, graph in
                create_chromosome_graphs(bedpe_df, -1)  # anchor adjustment = -1
            ]
            graphs.sort(key=lambda x: x[1])  # sort by index
            graphs_by_chromosome = {}
            for label, _, raw_label, graph in graphs:
                graphs_by_chromosome[label] = graph
                lab_str = label if label == raw_label else f'{label}("{raw_label}")'
                t.verbose(f'\t{lab_str}: |V|={graph.vcount()}, |E|={graph.ecount()}')
            # show examples
            example_graph = graphs[0][-1]        
            t.finished(f'Created {len(graphs_by_chromosome)} graphs.')
        self.printer.verbose(f'Example vertex: {example_graph.vs[0].attributes()}')
        self.printer.verbose(f'Example edge: {example_graph.es[0].attributes()}')
        return graphs_by_chromosome

    def cluster_fans(self, fans):

        def _get_fan_pos(fan):
            return fan.anchor_start

        order = get_sort_order(fans, _get_fan_pos)
        assert order != 'Unsorted'
        if order == 'Constant':
            return [list(fans)]

        return cluster_by_max_gap(fans, self.max_fan_cluster_gap, _get_fan_pos)

    def cluster_loops(self, fans):
        if len(fans) == 0:
            return []
        direction = fans[0].direction
        assert all(f.direction == direction for f in fans)
        fans = sorted(fans, key=lambda f: f.anchor_start)
        blades = itertools.chain.from_iterable(
            zip(f.positions, f.leafs, f.edges, f.edge_ids, itertools.repeat(f)) for f in fans
        )
        blades = direction.sort(blades, key=itemgetter(0))  # by position

        loop_clusters = [
            LoopCluster.from_blade_cluster(bc)
            for bc in cluster_by_max_gap(blades, self.max_loop_cluster_gap, itemgetter(0))
        ]
        return loop_clusters

    def extract_stripes(self, chromosome_graphs: Dict[str, ig.Graph]) -> List[Stripe]:
        
        def _make_stripes(graph_, fans, current_stripe_id, edge_to_stripe):
            fan_clusters = self.cluster_fans(fans)
            stripes = []
            for fan_cluster in fan_clusters:
                loop_clusters = self.cluster_loops(fan_cluster)
                stripe = Stripe(graph_, loop_clusters, stripe_id=current_stripe_id)
                for i, lc in enumerate(loop_clusters):
                    for eid in lc.edge_ids:
                        edge_to_stripe[eid] = (stripe, i)
                stripes.append(stripe)
                current_stripe_id += 1
            return stripes, current_stripe_id

        def _map_adjacent(stripes, e2s_parallel, e2s_orthogonal, g):
            for stripe in stripes:
                for i, lc in enumerate(stripe.loop_clusters):
                    for eid in lc.edge_ids:
                        if eid not in e2s_orthogonal:
                            es = g.es[eid]
                            raise RuntimeError('No orthogonal stripe')
                        orthogonal_stripe, j = e2s_orthogonal[eid]
                        stripe.orthogonal_stripes[i] = (orthogonal_stripe, j)
                        break  # TODO: this works?

        with self.timer('extract_stripes') as t:
            all_stripes = []
            chromosome_e2s = {}  # edge 2 stripe by chromosome
            stripe_id = 0
            for chromosome, graph in chromosome_graphs.items():
                right_fans, left_fans = fans_in_graph(graph)
                t.verbose(f'\tFans in {chromosome}: R={len(right_fans)}, L={len(left_fans)}')
                e2s_right = {}  # edge to (stripe, loop_cluster_idx)
                e2s_left = {}  # remember: edges (not vertices) are points on the heatmap!
                chromosome_e2s[chromosome] = (e2s_right, e2s_left)
                right_stripes, stripe_id = _make_stripes(graph, right_fans, stripe_id, e2s_right)
                left_stripes, stripe_id = _make_stripes(graph, left_fans, stripe_id, e2s_left)
                _map_adjacent(right_stripes, e2s_right, e2s_left, graph)
                _map_adjacent(left_stripes, e2s_left, e2s_right, graph)
                all_stripes.extend(right_stripes)
                all_stripes.extend(left_stripes)

            t.finished(f'Extracted {len(all_stripes)} raw stripes.')
            return all_stripes

    def _get_stripe_indices(self, stripes: List[Stripe]):
        stripe_sizes = [len(s) for s in stripes]
        offsets = np.cumsum([0] + stripe_sizes[:-1])
        stripe_sizes = np.array(stripe_sizes)
        return stripe_sizes, offsets
    
    def create_feature(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def calc_stripe_features(self, stripes: List[Stripe]):
        with self.timer('calc_stripe_features') as t:
            # Assign stripe data ranges
            stripe_sizes, offsets = self._get_stripe_indices(stripes)
            for i in range(len(stripes)):
                start = offsets[i]
                end = start + stripe_sizes[i]
                stripes[i].data_start = start
                stripes[i].data_end = end

            # Crerate feature buffers
            n_data_points = stripe_sizes.sum()
            self._stripe_id = self.create_feature(n_data_points, 'int64')
            self._f_length = self.create_feature(n_data_points, 'int64')
            self._f_loop_count = self.create_feature(n_data_points, 'int64')
            self._f_weight = self.create_feature(n_data_points, 'float64')
            self._f_gap_size = self.create_feature(n_data_points, 'int64')
            self._f_orientation_score = self.create_feature(n_data_points, 'float64')
            self._f_raw_gap_score = self.create_feature(n_data_points, 'float64')
            self._f_gap_score = self.create_feature(n_data_points, 'float64')
            self._f_stripe_score = self.create_feature(n_data_points, 'float64')
            self._f_worst_gap_score = self.create_feature(n_data_points, 'float64')
            self._f_l_score = self.create_feature(n_data_points, 'float64')
            self._f_w_score = self.create_feature(n_data_points, 'float64')
            self._f_cross_score = self.create_feature(n_data_points, 'float64')
            t.verbose('Created strripe feature arrays.')

            # Calculate basic features     
            for stripe in stripes:
                start = stripe.data_start
                end = stripe.data_end
                self._stripe_id[stripe.data_start:stripe.data_end] = stripe.id   # for debugging

                # Don't use Attrs.LENGTH - it will not be consistent.
                if stripe.direction.is_increasing:
                    base = stripe.anchor_start
                    prev = base
                    for i, lc in enumerate(stripe.loop_clusters, start):
                        self._f_length[i] = lc.leafs_end - base
                        self._f_gap_size[i] = lc.leafs_start - prev
                        prev = lc.leafs_end
                else:
                    base = stripe.anchor_end
                    prev = base
                    for i, lc in enumerate(stripe.loop_clusters, start):
                        self._f_length[i] = base - lc.leafs_start
                        self._f_gap_size[i] = prev - lc.leafs_end
                        prev = lc.leafs_start
                for i, lc in enumerate(stripe.loop_clusters, start):
                    self._f_weight[i] = sum(Attrs.WEIGHT.get_all(lc.edges))
                    self._f_loop_count[i] = sum(Attrs.LOOP_COUNT.get_all(lc.edges))
                    self._f_orientation_score[i] = lc.length / lc.width

                # Raw gap score
                _min_length = 1000.0
                amax = np.maximum
                gaps = self._f_gap_size[start:end]
                lengths = self._f_length[start:end]
                self._f_raw_gap_score[start:end] = amax(-np.log10(amax(gaps, _min_length) / amax(lengths, _min_length)), 0.0)
                # self._raw_gap_score[start] = np.nan

            # Feature quantiles
            quantile_levels = np.linspace(0.0, 1.0, num=self.n_quantile_levels + 2)
            quantiles = np.nanquantile(self._f_raw_gap_score, quantile_levels)
            unique_quantiles, idx = np.unique(quantiles, return_index=True)
            unique_quantile_levels = quantile_levels[idx]

            # Calculate features requiring distrributions estimated
            for stripe in stripes:
                start = stripe.data_start
                end = stripe.data_end
                self._stripe_id[stripe.data_start:stripe.data_end] = stripe.id

                # Gap score
                gap_scores = np.interp(self._f_raw_gap_score[start:end], unique_quantiles, unique_quantile_levels)
                self._f_gap_score[start:end] = gap_scores
                # self._gap_score[start] = 0.0
                self._f_worst_gap_score[start:end] = np.minimum.accumulate(gap_scores)
                
                # Stripe score
                q_scores = self._f_gap_score[start:end]
                n = len(q_scores)
                a = self.stripe_score_holder_exponent
                cum_means = np.cumsum(q_scores ** a) / np.arange(1, n + 1)
                self._f_stripe_score[start:end] = cum_means ** (1 / a)  # cumulative Holder mean

                # l, w and corss score
                score_feature = self._f_gap_score
                self_scores = score_feature[start:end]
                self_n = len(stripe.loop_clusters)
                for i, lc in enumerate(stripe.loop_clusters):
                    l_score = self_scores[i] if i == self_n - 1 else max(self_scores[i], self_scores[i + 1])
                    self._f_l_score[start + i] = l_score
                    ortho, j = stripe.orthogonal_stripes[i]
                    other_scores = score_feature.data[ortho.data_start:ortho.data_end]
                    ortho_n = len(ortho.loop_clusters)
                    w_score = other_scores[j] if j == ortho_n - 1 else min(other_scores[j], other_scores[j + 1])
                    self._f_w_score[start + i] = w_score
                _min_w_score = 0.01
                self._f_cross_score[start:end] = l_score / np.maximum(w_score, _min_w_score)
            
            # Assign features to stripes:
            for name in dir(self):
                if not name.startswith('_f_'):
                    continue
                data = getattr(self, name)
                for i, stripe in enumerate(stripes):
                    stripe.features[name[len('_f_'):]] = data[stripe.data_start:stripe.data_end].copy()


    def trim_stripe(self, stripe: Stripe, **kwargs) -> Stripe:
        if stripe.direction == 'R':
            return stripe
        _min_length = self.min_stripe_length
        _max_length = self.max_stripe_length if self.max_stripe_length != -1 else sys.maxsize - 1
        length: np.ndarray = stripe.features['length']
        worst_gap_score: np.ndarray = stripe.features['worst_gap_score']
        split_points = np.arange(len(length))
        k = -1
        for k in split_points:
            if length[k] < _min_length:
                continue
            if (
                    length[k] > _max_length or
                    worst_gap_score[k] < self.stripe_gap_threshold
            ):
                k -= 1
                break
        return stripe.trim(k + 1)

    def trim_stripes(self, stripes: List[Stripe], **kwargs) -> List[Stripe]:
        new_stripes = []
        with self.timer('trim_stripes') as t:
            # stripe_sizes, offsets = self._get_stripe_indices(stripes)
            for _, stripe in enumerate(stripes):
                new_stripe = self.trim_stripe(stripe)
                new_stripes.append(new_stripe)
            t.finished(f'Obtained {len(new_stripes)} trimmed stripes from {len(stripes)} raw stripes.')
        return new_stripes

    def filter_stripes(
            self, stripes,
            min_length=20_000,
            min_clusters=2,
            min_leafs=2,
            min_edegs=2,
            min_gap_score=0.0,
            min_stripe_score=0.0,
            min_cross_score=0.0,
            min_orientation_score=0.5
    ):
        filtered_stripes = []
        for s in stripes:
            if s.length < min_length:
                continue
            if len(s.loop_clusters) < min_clusters:
                continue
            if s.n_edges < min_edegs or s.n_leafs < min_leafs:
                continue
            if s.features['stripe_score'][-1] < min_stripe_score:
                continue
            if s.features['worst_gap_score'][-1] < min_gap_score:
                continue
            if s.features['cross_score'].mean() < min_cross_score:
                continue
            if s.features['orientation_score'].max() < min_orientation_score:
                continue
            filtered_stripes.append(s)
        return filtered_stripes

    def refine_stripes(self, all_stripes, max_length_diff=200_000):
        right_stripes = [s for s in all_stripes if s.direction == 'R']
        left_stripes = [s for s in all_stripes if s.direction == 'L']
        new_stripes = []
        for stripes in (left_stripes, right_stripes):
            prev: Optional[Stripe] = None
            for s in stripes:
                if prev and prev.chromosome != s.chromosome:
                    prev = None
                if prev:
                    if prev.anchor_end - s.anchor_start <= self.merge_stripes_threshold \
                        and s.anchor_end - prev.anchor_start <= self.max_merged_stripe_width \
                        and abs(s.length - prev.length) <= max_length_diff:
                        prev = prev.merge(s)
                    else:
                        new_stripes.append(prev)
                        prev = s
                else:
                    prev = s
            if prev:
                new_stripes.append(prev)
        return new_stripes

    @classmethod
    def get_stripes_dataframe(cls, stripes: List[Stripe]):
        df = pd.DataFrame.from_records(
            (
                s.chromosome, str(s.direction), s.id,
                s.anchor_start, s.anchor_end, s.width,
                s.start, s.end, s.length,
                s.n_fans, len(s.loop_clusters), s.n_leafs, s.n_edges,
                s.features['stripe_score'][-1],
                s.features['cross_score'].mean()
            )
            for s in stripes
        )
        df.columns = [
            'chromosome', 'direction', 'stripe_id',
            'anchor_start', 'anchor_end', 'width',
            'start', 'end', 'length',
            'n_fans', 'n_loop_clusters', 'n_leafs', 'n_edges',
            'stripe_score',
            'mean_cross_score'
        ]
        return df.set_index(['chromosome', 'direction', 'stripe_id'])

    def save_stripes(self, stripes: List[Stripe], output_file: str):
        df = self.get_stripes_dataframe(stripes)
        df.to_csv(output_file, sep='\t', quoting=csv.QUOTE_NONE)

    def save_stats(self, stripes: List[Stripe], output_dir: str):

        def _plot_file(_name):
            return os.path.join(output_dir, f'{_name}.png')

        def _describe(f) -> pd.Series:
            return pd.Series(f.data, copy=False, name=f.name).describe(percentiles=self.describe_percentiles)

        def _plot_distribution(f, plot_file: str, figsize=(8, 6), dpi=200):
            plt.figure(figsize=figsize, dpi=dpi)
            plt.hist(f.data, bins=50)
            plt.xlabel(f.name)
            plt.ylabel('Count')
            plt.savefig(plot_file)

        def _plot_against(f, x: np.ndarray, xlabel: str, plot_file: str, figsize=(8, 6), dpi=200):
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(x, f.data, 'b.', alpha=0.3)
            plt.xlabel(xlabel)
            plt.ylabel(f.name)
            plt.savefig(plot_file)

        with self.timer('analyze_features') as t:
            infos_table = pd.DataFrame([
                _describe(f) for f in self.features if not f.is_tagged('debug')
            ])
            t.print('Feature stats:\n' + infos_table.to_string(float_format='%.3f'))
            infos_table.to_csv(os.path.join(output_dir, 'feature_stats.csv'))
            length = self.features.length.data
            for f in self.features:
                if f.is_tagged('debug'):
                    continue
                _plot_distribution(f, _plot_file(f'dist_{f.name}'))
                if f.name != 'length':
                    _plot_against(f, length, 'length', _plot_file(f'len_{f.name}'))
            t.print('Created plots')

    def raw_pipeline(
            self,
            input_: Union[str, pd.DataFrame, Dict[str, ig.Graph]],
    ) -> List[Stripe]:
        loops_file = None
        bedpe_df = None
        graphs = None
        if isinstance(input_, str):
            loops_file = input_
        elif isinstance(input_, pd.DataFrame):
            bedpe_df = input_
        else:
            graphs = input_
        if graphs is None:
            with self.timer('processing raw interactions') as t:
                if bedpe_df is None:                
                    bedpe_df = read_raw_contacts_from_bedpe(
                        loops_file,
                        skiprows=self.skiprows,
                        petcount_col=self.petcount_col
                    )
                    t.finished(f'Read {len(bedpe_df)} intearctions from "{loops_file}"')
                bedpe_df = self.prepare_interactions(bedpe_df)
            graphs = self.create_graphs(bedpe_df)
        raw_stripes = self.extract_stripes(graphs)
        self.calc_stripe_features(raw_stripes)
        return raw_stripes

    def full_pipeline(
            self,
            input_: Union[str, pd.DataFrame, Dict[str, ig.Graph]]
    ) -> List[Stripe]:
        raw_stripes = self.raw_pipeline(input_)
        stripes = self.trim_stripes(raw_stripes)
        filtered_stripes = self.filter_stripes(
            stripes,
            min_length=self.min_stripe_length
        )
        refined_stripes = self.refine_stripes(filtered_stripes)
        return refined_stripes


def tuple_converter(elem_type=float, length=None, sep=','):
    def _tuple_converter(tuple_str):
        elem_strs = tuple_str.split(sep)
        if length is not None and len(elem_strs) != length:
            raise ValueError(f"Invalid length of tuple: actual={len(elem_strs)}, required={length}")
        result = tuple(elem_type(s.strip()) for s in elem_strs)
        return result

    return _tuple_converter


def main():
    parser = argparse.ArgumentParser(description="Call stripes using gStripe graph stripe-calling method.")
    parser.add_argument('input_file', help="Input bedpe file.")
    parser.add_argument('output_dir', help="Output dir.")
    parser.add_argument('--name', '-n', help="Output name (defaults to basename of input file.")
    parser.add_argument('--stats', '-s', action='store_true', help="Save statistics and plots to output directory?")
    parser.add_argument('--verbose', '-v', action='store_true', help="More verbose info (console log level=DEBUG)")
    parser.add_argument('--quiet', '-q', help="Suppress all console output")
    parser.add_argument('--loglevel', '-l', help="Logging level for file", default='DEBUG')
    add_argparse_args(GraphStripeCaller, parser)
    add_argparse_args(TaskRunner, parser)
    args = parser.parse_args()

    output_dir = args.output_dir
    input_file = args.input_file
    output_name = args.name if args.name else os.path.basename(input_file)
    log_file = os.path.join(output_dir, f'{output_name}.gstripe.log')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    logger = logging.getLogger(__name__)
    console_level = 'INFO'
    if args.quiet:
        console_level = 'NOTSET'
    elif args.verbose:
        console_level = 'DEBUG'
    setup_logger(
        logger,
        console_level=console_level,
        file_level=args.loglevel,
        log_file=log_file
    )
    printer = MessagePrinter(quiet=args.quiet, verbose=args.verbose, logger=logger)

    runner: TaskRunner = from_argparse_args(TaskRunner, args, logger=logger, printer=printer)
    caller: GraphStripeCaller = from_argparse_args(GraphStripeCaller, args, logger=logger, printer=printer)
    with runner, Timer('main', printer) as t:
        results_file = os.path.join(output_dir, f'{output_name}.gstripes_raw.tsv')
        stripes = caller.full_pipeline(input_file)
        t.message(f'Called {len(stripes)} stripes.')
        caller.save_stripes(stripes, results_file)        
        t.message(f'Results saved to "{results_file}"')
        if args.stats:
            stats_dir = os.path.join(output_dir, output_name)
            t.message(f'Saving stats to "{stats_dir}"')
            if not os.path.exists(stats_dir):
                os.mkdir(stats_dir)
            caller.save_stats(stripes, stats_dir)
        t.finished(f'All done.')


if __name__ == '__main__':
    main()
