import csv
import math
from itertools import repeat
from typing import Optional, Union, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import hicstraw
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from gstripe.gstripe import Stripe, GraphStripeCaller
from gstripe.genomic_tools import Region
from pandas import DataFrame, Series

from .ranges import RangeSeries, overlapping_pairs_grouped

JUICER_COORD_COLS = ['chr', 'pos1', 'pos2', 'chr2', 'pos3', 'pos4']


def read_stripes_gstripe(path):
    df = pd.read_csv(path, sep='\t').drop(columns='color')
    df['chr'] = ('chr' + df['chr']).astype(HumanChromosomeDtype)
    df['chr2'] = ('chr' + df['chr2']).astype(HumanChromosomeDtype)
    df = df.sort_values(JUICER_COORD_COLS).reset_index(drop=True)
    df.index.names = ['stripe_id']
    return df


HumanChromosomeDtype = pd.CategoricalDtype([
    f'chr{i}' for i in list(range(1, 22 + 1)) + ['X', 'Y', 'M']
], ordered=True)


def read_juicebox_stripes(path, index_col=None):
    df = pd.read_csv(path, sep='\t')
    df.index.names = ['stripe_id']
    df = df.reset_index()
    is_right = (df.pos3 - df.pos1).abs() < (df.pos4 - df.pos2).abs()
    res_df = DataFrame({
        'stripe_id': df.stripe_id,
        'chromosome': ('chr' + df.chr.astype(str)).astype(HumanChromosomeDtype),
        'direction': np.where(is_right, 'R', 'L'),
        'anchor_start': df.pos1,
        'anchor_end': df.pos2,
        'width': df.pos2 - df.pos1,
        'start': df.pos3,
        'end': df.pos4,
        'length': df.pos4 - df.pos3
    })
    if index_col is not None:
        res_df = res_df.set_index(index_col)
    return res_df


def save_juicebox_stripes(df, path, color='#009900', pad_w=0, pad_l=0):
    """
    From StripeNN docs:
    chromosome1 x1 x2 chromosome2 y1 y2 color
    chromosome = the chromosome that the domain is located on
    x1,x2/y1,y2 = the interval spanned by the domain (contact domains manifest as squares on the diagonal of a Hi-C matrix and as such: x1=y1, x2=y2)
    color = the color that the values will be rendered as if loaded in Juicebox
    """
    R = df.direction == 'R'
    chromosome = df.chromosome.str[3:]
    jdf = DataFrame({
        'chr': chromosome,
        'pos1': df.anchor_start - pad_w,
        'pos2': df.anchor_end + pad_w,
        'chr2': chromosome,
        'pos3': df.start - np.where(R, pad_l, pad_w),
        'pos4': df.end + np.where(~R, pad_l, pad_w),
        'color': color,
        'stripe_id': df.stripe_id,
        'direction': df.direction
    })
    # flip right stripes:
    # juicer_df.loc[R, juicer_df.columns[:6]] = juicer_df.loc[R, juicer_df.columns[[3, 4, 5, 0, 1, 2]]]
    jdf.loc[R, 'pos1'] -= 1  # adjustment for juicebox plotting
    jdf.to_csv(path, index=False, sep='\t', quoting=csv.QUOTE_NONE)


def save_stripe_domains(df, path, pad_w=0, pad_l=0):
    R = df.direction == 'R'
    bed_df = DataFrame({
        'chr': df.chromosome,
        'start': df.start - np.where(R, pad_l, pad_w),
        'end': df.end + np.where(~R, pad_l, pad_w)
    })
    bed_df.to_csv(path, index=False, sep='\t', quoting=csv.QUOTE_NONE, header=False)


def read_stripes_stripenn(path):
    df = pd.read_csv(
        path, sep='\t',
        dtype={'pos1': int, 'pos2': int, 'pos3': int, 'pos4': int, 'length': int, 'width': int}
    )
    df['chr'] = ('chr' + df['chr'].astype(str)).astype(HumanChromosomeDtype)
    df['chr2'] = ('chr' + df['chr2'].astype(str)).astype(HumanChromosomeDtype)
    df = df.sort_values(JUICER_COORD_COLS).reset_index(drop=True)
    df.index.names = ['stripe_id']
    df.insert(6, 'direction', np.where((df.pos2 + df.pos1) - (df.pos4 + df.pos3) > 0, 'L', 'R'))
    return df


# def write_jucebox_files(data, target_dir, prefix='', suffix='', colors=('#ff0000', '#00ff00')):
#     for idx, stripes in data.groupby(data.index.names[:-1]):
#         outfile_name = prefix + '_'.join(map(str, idx)) + f'{suffix}.tsv'
#         if len(stripes) == 0:
#             continue
#         stripes = stripes.sort_values(JUICER_COORD_COLS)
#         if str(stripes.chr.iat[0]).startswith('chr'):
#             stripes['chr'] = stripes['chr'].str[3:]
#             stripes['chr2'] = stripes['chr2'].str[3:]
#         if 'direction' not in stripes.columns:
#             stripes['direction'] = np.where(stripes.pos1 == stripes.pos3, 'R', 'L')
#         stripes['color'] = np.where(stripes.direction == 'R', *colors)
#         stripes['pos1'] = np.where(stripes.direction == 'R', stripes.pos1 - 1, stripes.pos1)  # adjustment for juicebox plotting
#         stripes = stripes.loc[:, JUICER_COORD_COLS + ['color', 'direction']]
#         stripes.to_csv(os.path.join(target_dir, outfile_name), index=False, sep='\t')


def add_suffix(strings, suffix):
    if isinstance(strings, str):
        return f'{strings}{suffix}'
    return [f'{s}{suffix}' for s in as_list(strings)]


def map_suffix(strings, suffix):
    return dict(zip(strings, add_suffix(strings, suffix)))


def as_list(x, scalar_type=str):
    if x is None:
        return []
    elif isinstance(x, scalar_type):
        return [x]
    else:
        return list(x)


def _as_list_AB(x_A, x_B):
    if x_B is None:
        x_B = x_A
    return as_list(x_A), as_list(x_B)


def range_join(
        df_A: DataFrame,
        df_B: DataFrame,
        ranges_A: RangeSeries,
        ranges_B: RangeSeries,
        on_A: Union[str, List[str], None],
        on_B: Union[str, List[str], None] = None,
        tolerance: int = 0,
        suffixes: Tuple[str, str] = ('_A', '_B'),
        overlap_col: Optional[str] = 'overlap'
):
    # TODO: 0 length
    on_A, on_B = _as_list_AB(on_A, on_B)
    _A, _B = suffixes
    if not all(c in ranges_A.index.names for c in on_A):
        raise ValueError(f'Not all groups_A({on_A}) in index of ranges_A: {ranges_A.index.names}')
    if not all(c in ranges_B.index.names for c in on_B):
        raise ValueError(f'Not all groups_B({on_B}) in index of ranges_B: {ranges_B.index.names}')

    df_A = df_A.reset_index()
    df_B = df_B.reset_index()

    common_cols = set(df_A.columns).intersection(df_B.columns)
    # for col in on_B:
    #     common_cols.remove(col)
    df_A = df_A.rename(columns=map_suffix(common_cols, _A))
    df_B = df_B.rename(columns=map_suffix(common_cols, _B))
    # _to_drop_in_B = [col for col in on_B if col not in common_cols]

    if tolerance != 0:
        ranges_A = ranges_A.expand(tolerance // 2 + tolerance % 2)
        ranges_B = ranges_B.expand(tolerance // 2)

    idx = overlapping_pairs_grouped(ranges_A, ranges_B, on_A, on_B)
    idx_A = idx[:, 0]
    idx_B = idx[:, 1]
    del idx
    merged_df = pd.concat([
        df_A.iloc[idx_A].reset_index(drop=True),
        df_B.iloc[idx_B].reset_index(drop=True)  # TODO:  drop(columns=_to_drop_in_B)
    ], axis=1)

    if overlap_col:
        assert overlap_col not in merged_df.columns
        r_A = ranges_A.subset(idx_A).reset_index()
        r_B = ranges_B.subset(idx_B).reset_index()
        merged_df[overlap_col] = -(r_A.offset(r_B) + tolerance)

    return merged_df


def merge_ranges(
        df_A: DataFrame,
        df_B: DataFrame,
        ranges_A: RangeSeries,
        ranges_B: RangeSeries,
        by_A: Union[str, List[str], None] = None,
        by_B: Union[str, List[str], None] = None,
        on_A: Union[str, List[str], None] = None,
        on_B: Union[str, List[str], None] = None,
        inner_on_A: Union[str, List[str], None] = None,
        inner_on_B: Union[str, List[str], None] = None,
        ids_A: Optional[str] = None,
        ids_B: Optional[str] = None,
        tolerance: int = 0,
        suffixes: Tuple[str, str] = ('_A', '_B'),
        overlap_col: Optional[str] = 'overlap',
        assume_sorted: bool = False
):
    # Prepare inputs
    by_A, by_B = _as_list_AB(by_A, by_B)
    on_A, on_B = _as_list_AB(on_A, on_B)
    if not len(on_A) == len(on_B):
        raise ValueError(f'Outer matching groups must have the same number of levels, instead A={on_A}, B={on_B}')
    inner_on_A, inner_on_B = _as_list_AB(inner_on_A, inner_on_B)
    if not len(inner_on_A) == len(inner_on_B):
        raise ValueError(
            f'Inner matching groups must have the same number of levels, instead A={inner_on_A}, B={inner_on_B}')
    _A, _B = suffixes

    if not assume_sorted:
        df_A = df_A.sort_values(by_A + on_A + inner_on_A)
        ranges_A = ranges_A(df_A)
        df_B = df_B.sort_values(by_B + on_B + inner_on_B)
        ranges_B = ranges_B(df_B)

    def _prepare_df(df, by, ids, suff):
        if ids is None:
            raise NotImplementedError()
        _to_reset = map_suffix(by + [ids], suff)
        df = df.reset_index([col for col in _to_reset.keys() if col in df.index.names])  # TODO: what if only index?
        for col in _to_reset.keys():
            if not col in df.columns:
                raise ValueError(f'Column {col} not found in df{suff} with columns {df.columns}')
        df = df.rename(columns=_to_reset)
        return df, _to_reset[ids]

    df_A, ids_A = _prepare_df(df_A, by_A, ids_A, _A)
    df_B, ids_B = _prepare_df(df_B, by_B, ids_B, _B)

    _full_on_A = on_A + inner_on_A
    _full_on_B = on_B + inner_on_B
    merged_df = range_join(
        df_A, df_B,
        ranges_A, ranges_B,
        _full_on_A, _full_on_B,
        tolerance,
        suffixes,
        overlap_col=overlap_col
    )
    return merged_df


def merge_ranges_single(
        df: DataFrame,
        ranges: RangeSeries,
        by: Union[str, List[str], None],
        on: Union[str, List[str], None],
        inner_on: Union[str, List[str], None],
        ids: Optional[str],
        tolerance: int = 0,
        suffixes: Tuple[str, str] = ('_A', '_B'),
        overlap_col: Optional[str] = 'overlap',
        assume_sorted: bool = False
):
    on = as_list(on)
    inner_on = as_list(inner_on)
    if not assume_sorted:
        df = df.sort_values(on + inner_on + list(ranges.names))
        ranges = ranges(df)
    merged_df = merge_ranges(
        df, df,
        ranges, ranges,
        by, by,
        on, on,
        inner_on, inner_on,
        ids, ids,
        tolerance=tolerance,
        suffixes=suffixes,
        overlap_col=overlap_col,
        assume_sorted=True
    )
    return merged_df


def count_overlaps(
        merged_df: DataFrame,
        df_A: DataFrame,
        df_B: DataFrame,
        by_A: Union[str, List[str], None] = None,
        by_B: Union[str, List[str], None] = None,
        on_A: Union[str, List[str], None] = None,
        on_B: Union[str, List[str], None] = None,
        inner_on_A: Union[str, List[str], None] = None,
        inner_on_B: Union[str, List[str], None] = None,
        ids_A: Optional[str] = None,
        ids_B: Optional[str] = None,
        min_overlap: int = 0,
        overlap_col: Optional[str] = 'overlap',
        suffixes: Tuple[str, str] = ('_A', '_B')
):
    # Prepare inputs
    by_A, by_B = _as_list_AB(by_A, by_B)
    on_A, on_B = _as_list_AB(on_A, on_B)
    if not len(on_A) == len(on_B):
        raise ValueError(f'Outer matching groups must have the same number of levels, instead A={on_A}, B={on_B}')
    inner_on_A, inner_on_B = _as_list_AB(inner_on_A, inner_on_B)
    if not len(inner_on_A) == len(inner_on_B):
        raise ValueError(
            f'Inner matching groups must have the same number of levels, instead A={inner_on_A}, B={inner_on_B}')
    _A, _B = suffixes

    # Count sizes
    def _count(df, by, on, ids, suff):
        count_df = df.groupby(add_suffix(by + on, suff), observed=False)[ids].nunique().to_frame(add_suffix('n', suff)).copy()
        count_df.index.names = add_suffix(count_df.index.names, suff)
        return count_df

    counts_A = _count(df_A, by_A, ids_A, _A)
    counts_B = _count(df_B, by_B, ids_B, _B)

    by_AB = add_suffix(by_A, _A) + add_suffix(by_B, _B)
    grouping = merged_df[merged_df[overlap_col] >= min_overlap].groupby(by_AB, observed=False)

    # Prepare result dataframe
    res = DataFrame({
        add_suffix('m', _A): grouping[ids_A].nunique(),
        add_suffix('m', _B): grouping[ids_B].nunique()
    })
    res = res.reset_index()
    res = pd.merge(res, counts_A.reset_index())
    res = pd.merge(res, counts_B.reset_index())
    res[add_suffix('p', _A)] = res[add_suffix('m', _A)] / res['n' + _A]
    res[add_suffix('p', _B)] = res[add_suffix('m', _B)] / res['n' + _B]
    return res.set_index(by_AB)


def count_overlaps_single(
        merged_df: DataFrame,
        df: DataFrame,
        by: Union[str, List[str], None] = None,
        on: Union[str, List[str], None] = None,
        # inner_on: Union[str, List[str], None] = None,
        ids: Optional[str] = None,
        min_overlap: int = 0,
        overlap_col: Optional[str] = 'overlap',
        suffixes: Tuple[str, str] = ('_A', '_B')
):
    # Prepare inputs
    by = as_list(by)
    on = as_list(on)
    # inner_on = as_list(inner_on)
    _A, _B = suffixes

    counts = df.groupby(by + on, observed=False)[ids].nunique()
    counts_A = counts.to_frame(add_suffix('n', _A)).copy()
    counts_A.index.names = add_suffix(counts_A.index.names, _A)
    counts_B = counts.to_frame(add_suffix('n', _B)).copy()
    counts_B.index.names = add_suffix(counts_B.index.names, _B)

    by_AB = add_suffix(on + by, _A) + add_suffix(on + by, _B)
    grouping = merged_df[merged_df[overlap_col] >= min_overlap].groupby(by_AB, observed=False)

    # Prepare result dataframe
    res = DataFrame({
        add_suffix('m', _A): grouping[add_suffix(ids, _A)].nunique(),
        add_suffix('m', _B): grouping[add_suffix(ids, _B)].nunique()
    })
    res = res.reset_index()
    res = pd.merge(res, counts_A.reset_index())
    res = pd.merge(res, counts_B.reset_index())
    res[add_suffix('p', _A)] = res[add_suffix('m', _A)] / res['n' + _A]
    res[add_suffix('p', _B)] = res[add_suffix('m', _B)] / res['n' + _B]
    return res.set_index(by_AB)


def stripes_dataframe(stripes: Union[DataFrame, List[Stripe]]):
    if isinstance(stripes, DataFrame):
        return stripes
    elif isinstance(stripes, Series):  # Series of lists of Stripe objects
        df = pd.concat({
            idx: GraphStripeCaller.get_stripes_dataframe(data)
            for idx, data in stripes.items()
        }, names=stripes.index.names)
    else:  # Single list of stripe objects
        df = GraphStripeCaller.get_stripes_dataframe(stripes)
    return df


def stripe_regions(stripes, coord_cols=('anchor_start', 'anchor_end')):
    df = stripes_dataframe(stripes)
    ranges = RangeSeries(df.loc[:, coord_cols[0]], df.loc[:, coord_cols[1]])
    return df, ranges


def mapping_table(
        comparison_df: DataFrame,
        by_A: Union[str, List[str], None] = None,
        by_B: Union[str, List[str], None] = None,
        on_A: Union[str, List[str], None] = None,
        on_B: Union[str, List[str], None] = None,
        prop: bool = False,
        suffixes: Tuple[str, str] = ('_A', '_B'),
        A_to_B: bool = True,
):
    by_A, by_B = _as_list_AB(by_A, by_B)
    on_A, on_B = _as_list_AB(on_A, on_B)
    _A, _B = suffixes
    col = add_suffix('p' if prop else 'm', _A if A_to_B else _B)
    row_groups = add_suffix(on_A + by_A, _A)
    col_groups = add_suffix(by_B, _B)
    n_data_columns = 6
    to_keep = set(comparison_df.columns[-n_data_columns:])
    to_keep.update(row_groups)
    to_keep.update(col_groups)
    to_drop = [c for c in comparison_df.columns if c not in to_keep]
    tab = comparison_df.drop(columns=to_drop).set_index(
        row_groups + col_groups
    ).loc[:, col].unstack(col_groups, fill_value=np.nan if prop else 0)
    return tab


# def mapping_table(
#         comparison_df: DataFrame,
#         groups_A: Union[str, List[str]],
#         groups_B: Union[str, List[str], None] = None,
#         prop: bool = False,
#         suffixes: Tuple[str, str] = ('_A', '_B'),
#         A_to_B: bool = True,
# ):
#     _A, _B = suffixes
#     col = ('p' if prop else 'm') + (_A if A_to_B else _B)
#     groups_A, groups_B = _as_list_AB(groups_A, groups_B)
#     groups_A = add_suffix(groups_A, _A)
#     groups_B = add_suffix(groups_B, _B)
#     n_data_columns = 6
#     to_keep = set(comparison_df.columns[-n_data_columns:])
#     to_keep.update(groups_A)
#     to_keep.update(groups_B)
#     to_drop = [c for c in comparison_df.columns if c not in to_keep]
#     tab = comparison_df.drop(columns=to_drop).set_index(
#         groups_A + groups_B
#     ).loc[:, col].unstack(groups_B)
#     return tab


def mapping_heatmap(
        comparison_df: DataFrame,
        by_A: Union[str, List[str], None] = None,
        by_B: Union[str, List[str], None] = None,
        on_A: Union[str, List[str], None] = None,
        on_B: Union[str, List[str], None] = None,
        suffixes: Tuple[str, str] = ('_A', '_B'),
        cmap='coolwarm_r',
        legend=True,
        dataset='',
        A_to_B=True,
        exclude_diagonal=None,
        ax: Optional[plt.Axes] = None
):
    by_A, by_B = _as_list_AB(by_A, by_B)
    on_A, on_B = _as_list_AB(on_A, on_B)
    _A, _B = suffixes
    counts = mapping_table(comparison_df, by_A, by_B, on_A, on_B, prop=False, suffixes=suffixes, A_to_B=A_to_B)
    props = mapping_table(comparison_df, by_A, by_B, on_A, on_B, prop=True, suffixes=suffixes, A_to_B=A_to_B)
    if ax is None:
        ax = plt.gca()
    if exclude_diagonal is None:
        if isinstance(counts.index, pd.MultiIndex) == isinstance(counts.columns, pd.MultiIndex):
            if isinstance(counts.columns, pd.MultiIndex):
                exclude_diagonal = (counts.index == counts.columns).all()
            else:
                exclude_diagonal = list(counts.index) == list(counts.columns)  # TODO: check
        else:
            exclude_diagonal = False
    data = props * 100
    if not A_to_B:
        data = data.transpose()
    annot = np.round(data.fillna(0), 0).astype(int)
    if exclude_diagonal:
        if len(on_A) == 0:
            assert len(on_B) == 0
            np.fill_diagonal(data.values, np.nan)
        else:
            grps = add_suffix(on_A[0] if len(on_A) == 1 else on_A, _A)
            for grp_idx, _ in data.groupby(grps, observed=False):
                np.fill_diagonal(data.loc[grp_idx].values, np.nan)
    sns.heatmap(
        data,
        annot=annot,
        vmin=0, vmax=100,
        linewidths=.5,
        cmap=cmap,
        fmt=',d',
        square=True,
        ax=ax,
        cbar=legend,
        annot_kws={
            'fontsize': 'medium'
        }
    )
    if exclude_diagonal:
        _totals = np.diag(counts.values)
        for i in range(len(data)):
            ax.text(i + 0.5, i + 0.5, f'{_totals[i]:,d}', ha='center', va='center',
                    fontsize='small', fontweight='bold')
    if dataset:
        dataset += ': '
    _title = f'{dataset} % stripes in row mapped to column.'
    ax.set_title(_title)
    return ax


def compare_stripes_single(
        stripes, by, on,
        tolerance=0,
        ids='stripe_id', inner_on=('chromosome', 'direction'),
        by_types=None,
        ax=None,
        legend=True
):
    sdf, sr = stripe_regions(stripes)
    inner_on = as_list(inner_on)
    mdf = merge_ranges_single(sdf, sr, by=by, on=on, inner_on=inner_on, ids=ids, tolerance=tolerance)
    comp_df = count_overlaps_single(mdf, sdf.reset_index(), by=by, on=on, ids=ids)
    cnt_tab = mapping_table(comp_df.reset_index(), by_A=by, on_A=on, prop=False)
    if ax is None:
        ax = plt.gca()
    if by_types is not None:
        idxs = list(comp_df.index.names)
        comp_df = comp_df.reset_index()
        for _name, _type in zip(idxs, by_types):
            comp_df[_name] = comp_df[_name].astype(_type)
        comp_df = comp_df.set_index(idxs)    
    comp_df = comp_df.sort_index()
    heatmap_ax = mapping_heatmap(
        comp_df.reset_index(), by_A=by, on_A=on, exclude_diagonal=False, ax=ax, legend=legend
    )
    return mdf, comp_df, cnt_tab, heatmap_ax


def compare_striping_domains_single(
        stripes, by, on,
        tolerance=0,
        ids='stripe_id', inner_on=('chromosome', 'direction'),
        by_types=None,
        ax=None,
        legend=True
):
    by = as_list(by)
    on = as_list(on)    
    inner_on = as_list(inner_on)
    cc = ['start', 'end']
    sdf0, sr0 = stripe_regions(stripes, coord_cols=cc)
    sdf0 = sdf0.sort_values(on + inner_on + cc)
    sr0 = sr0(sdf0)    
    sr = sr0.groupby(by + on + inner_on, observed=False).union_self()
    new_index = pd.MultiIndex.from_frame(
        sr.index.to_frame(index=False).rename_axis(ids).reset_index())
    sr = sr.set_index(new_index)
    sdf = sr.to_frame()
    mdf = merge_ranges_single(sdf, sr, by=by, on=on, inner_on=inner_on, ids=ids, tolerance=tolerance)
    comp_df = count_overlaps_single(mdf, sdf.reset_index(), by=by, on=on, ids=ids)
    cnt_tab = mapping_table(comp_df.reset_index(), by_A=by, on_A=on, prop=False)
    if ax is None:
        ax = plt.gca()
    if by_types is not None:        
        idxs = list(comp_df.index.names)        
        comp_df = comp_df.reset_index()
        for _name, _type in zip(idxs, by_types):
            comp_df[_name] = comp_df[_name].astype(_type)
        comp_df = comp_df.set_index(idxs)
    heatmap_ax = mapping_heatmap(
        comp_df.reset_index(), by_A=by, on_A=on, exclude_diagonal=False, ax=ax, legend=legend
    )
    return mdf, comp_df, cnt_tab, heatmap_ax


def collapse_dataset_indexes(df, map_=None, omit=None, sep='_', suffixes=('_A', '_B'), name='dataset'):
    columns = [
        [
            col for col in df.index.names
            if col.endswith(suf) and (omit is None or not col.startswith(omit))
        ]
        for suf in suffixes
    ]
    df = df.reset_index(columns[0] + columns[1])
    joined = [
        Series(
            df.loc[:, columns[i]].applymap(str).values.tolist(),
            index=df.index
        ).map(tuple)
        for i in range(2)
    ]
    if map_ is not None:
        if isinstance(map_, str):
            map_ = lambda x: sep.join(x)
        joined = [col.map(map_) for col in joined]
    df = df.drop(columns=columns[0] + columns[1])
    new_names = [f'{name}{suf}' for suf in suffixes]
    for col_vals, col_name in zip(joined, new_names):
        df[col_name] = col_vals
    return df.set_index(new_names, append=True)


def plot_stripe_stats(stripes, by_direction=True, dist=('length', 'width'), title=''):
    stripes = stripes_dataframe(stripes)
    if not stripes.stripe_id.is_unique:
        print('Warning, nonunique stripe ids.')
    grp = 'direction' if by_direction else None
    n_small = len(dist)
    n_rows = 1 + n_small // 2 + n_small % 2
    fig = plt.figure(figsize=(8, n_rows * 3), constrained_layout=True)
    gs = GridSpec(n_rows, 2, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    p = sns.countplot(data=stripes, x='chromosome', hue=grp, ax=ax)
    p.set_xticklabels(p.get_xticklabels(), rotation=45)
    for i, var in enumerate(dist):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        sns.kdeplot(data=stripes, x=var, hue=grp, ax=ax)
    fig.suptitle(f'{title} (N={len(stripes)})')
    fig.tight_layout()


def _repeat_if_not_vector(x):
    if hasattr(x, '__len__') and not (isinstance(x, str) or isinstance(x, tuple)):
        return x
    return repeat(x)


def _mirror_diagonally(x, y, which=None):
    if which is None:
        return y, x
    else:
        xt = np.where(which, y, x)
        yt = np.where(which, x, y)
        return xt, yt


class HiCHeatmap(object):
    def __init__(
            self,
            data,
            xlim: Union[Region, Tuple[str, int, int]],
            ylim: Union[Region, Tuple[str, int, int], None] = None,
            resolution: Optional[int] = None,
            data_type: str = 'observed',
            normalization: str = "NONE"
    ):
        if isinstance(xlim, tuple):
            xlim = Region(*xlim)
        if ylim is None:
            ylim = xlim
        elif isinstance(ylim, tuple):
            ylim = Region(*ylim)
        self.xlim = xlim
        self.ylim = ylim
        self.x_chromosome = None
        self.y_chromosome = None
        if not isinstance(data, np.ndarray):
            import hicstraw
            if isinstance(data, str):
                data = hicstraw.HiCFile(data)
            if resolution is None:
                resolution = min(data.getResolutions())
            self.resolution = resolution
            self.x_chromosome = xlim.chromosome
            self.y_chromosome = ylim.chromosome
            mzd = data.getMatrixZoomData(
                self.x_chromosome[3:], self.y_chromosome[3:],
                data_type, normalization, "BP", self.resolution
            )
            self.matrix_data = mzd.getRecordsAsMatrix(xlim.start, xlim.end, ylim.start, ylim.end)
        else:
            if resolution is None:
                raise ValueError("Must specify resolution if hic file not provided")
            self.resolution = resolution
            self.matrix_data = data
        self.ax: Optional[matplotlib.pyplot.Axes] = None

    def heatmap(self, cmap=None, max_color=0.95, ax=None):
        if ax is None:
            ax: matplotlib.pyplot.Axes = plt.gca()
        self.ax = ax
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])
        if isinstance(max_color, float):
            max_color = max(np.quantile(self.matrix_data, max_color), 1)
        self.ax.imshow(self.matrix_data, cmap=cmap, vmin=0, vmax=max_color, aspect='equal')

    def _transform_x(self, x):
        return (x - self.xlim.start) / self.resolution

    def _transform_y(self, y):
        return (y - self.ylim.start) / self.resolution

    def points(self, x, y, where='asis', color='black', size=3.0, marker='o', alpha=None):
        if self.ax is None:
            self.ax = plt.gca()
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        assert x.shape == y.shape
        assert where.lower() in ('upper', 'u', 'lower', 'l', 'both', 'b', 'asis', 'a')

        if where.lower().startswith('u'):
            x, y = _mirror_diagonally(x, y, y > x)
        elif where.lower().startswith('l'):
            x, y = _mirror_diagonally(x, y, y < x)
        elif where.lower().startswith('b'):
            xu, yu = _mirror_diagonally(x, y, y > x)
            xl, yl = _mirror_diagonally(x, y, y < x)
            x = np.concatenate([xu, xl])
            y = np.concatenate([yu, yl])

        sel = (x >= self.xlim.start) & (x <= self.xlim.end) & (y >= self.ylim.start) & (y <= self.ylim.end)

        self.ax.scatter(
            self._transform_x(x[sel]), self._transform_y(y[sel]),
            s=size, c=color, edgecolors='none', marker=marker, alpha=alpha
        )

    def rectangles(self, x0, x1, y0, y1, where='asis', color='black', linewidth=1.0, fill=False, alpha=None):
        if self.ax is None:
            self.ax = plt.gca()
        color = _repeat_if_not_vector(color)
        assert where.lower() in ('upper', 'u', 'lower', 'l', 'both', 'b', 'asis', 'a')
        if isinstance(x0, list):
            x0 = np.array(x0)
        if isinstance(x1, list):
            x1 = np.array(x1)
        if isinstance(y0, list):
            y0 = np.array(y0)
        if isinstance(y1, list):
            y1 = np.array(y1)

        assert (x0 <= x1).all()
        assert (y0 <= y1).all()

        if not where.lower().startswith('a'):
            is_below = (y0 - x1).abs() < (y1 - x0).abs()  # upper right < lower left
            if where.lower().startswith('u'):
                x0, y0 = _mirror_diagonally(x0, y0, is_below)
                x1, y1 = _mirror_diagonally(x1, y1, is_below)
            elif where.lower().startswith('l'):
                x0, y0 = _mirror_diagonally(x0, y0, ~is_below)
                x1, y1 = _mirror_diagonally(x1, y1, ~is_below)
            elif where.lower().startswith('b'):
                ux0, uy0 = _mirror_diagonally(x0, y0, is_below)
                ux1, uy1 = _mirror_diagonally(x1, y1, is_below)
                lx0, ly0 = _mirror_diagonally(x0, y0, ~is_below)
                lx1, ly1 = _mirror_diagonally(x1, y1, ~is_below)
                x0 = np.concatenate([ux0, lx0])
                x1 = np.concatenate([ux1, lx1])
                y0 = np.concatenate([uy0, ly0])
                y1 = np.concatenate([uy1, ly1])

        sel = (x0 <= self.xlim.end) & (x1 >= self.xlim.start) & \
              (y0 <= self.ylim.end) & (y1 >= self.ylim.start)

        x_starts = self._transform_x(x0[sel])
        y_starts = self._transform_y(y0[sel])
        x_ends = self._transform_x(x1[sel])
        y_ends = self._transform_y(y1[sel])

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        patches = [
            Rectangle(
                (xs, ys), w, h,
                color=c, linewidth=linewidth, fill=fill, alpha=alpha
            )
            for xs, ys, w, h, c in zip(
                x_starts,
                y_starts,
                x_ends - x_starts,
                y_ends - y_starts,
                color
            )
        ]
        for patch in patches:
            self.ax.add_patch(patch)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def line(self, x0, y0, x1, y1, color='black', linewidth=1.0, alpha=None):
        if self.ax is None:
            self.ax = plt.gca()
        xdata = self._transform_x(x0), self._transform_x(x1)
        ydata = self._transform_y(y0), self._transform_y(y1)
        self.ax.add_line(Line2D(
            xdata, ydata, linewidth=linewidth, color=color
        ))

    def text(self, x, y, text, **kwargs):
        if self.ax is None:
            self.ax = plt.gca()
        if x > self.xlim.end or x < self.xlim.start or y > self.ylim.end or y < self.ylim.start:
            return
        xt = self._transform_x(x)
        yt = self._transform_y(y)
        self.ax.text(xt, yt, text, **kwargs)

    def regions2d(self, region_x: RangeSeries, region_y: RangeSeries, where='asis',
                  color='black', linewidth=1.0, fill=False, alpha=None,
                  chromosome_name_A='chromosome_A', chromosome_name_B='chromosome_B'):
        if self.x_chromosome is not None:
            sel = (region_x.index.get_level_values(chromosome_name_A) == self.x_chromosome) & \
                  (region_y.index.get_level_values(chromosome_name_B) == self.y_chromosome)
            region_x = region_x.subset(sel)
            region_y = region_y.subset(sel)
        self.rectangles(
            region_x.start, region_x.end, region_y.start, region_y.end, where,
            color, linewidth, fill, alpha
        )

    def loops(
            self, loops: DataFrame, where='asis',
            color='blue', size=1.0, marker='o', alpha=None,
            chromosome_name='chromosome', suffixes=('_A', '_B')
    ):
        chromosome_name_A, chromosome_name_B = [f'{chromosome_name}{s}' for s in suffixes]
        for col in (chromosome_name_A, chromosome_name_B):
            if col not in loops.index.names:
                loops = loops.set_index(col, append=True)
        anchor_A = RangeSeries(loops, suffix=suffixes[0])
        anchor_B = RangeSeries(loops, suffix=suffixes[1])
        self.loop_anchors(
            anchor_A, anchor_B, where,
            color, size, marker, alpha,
            chromosome_name_A, chromosome_name_B
        )

    def loop_anchors(
            self, anchor_A: RangeSeries, anchor_B: RangeSeries, where='asis',
            color='blue', size=1.0, marker='o', alpha=None,
            chromosome_name_A='chromosome_A', chromosome_name_B='chromosome_B'
    ):
        if self.x_chromosome is not None:
            sel = (anchor_A.index.get_level_values(chromosome_name_A) == self.x_chromosome) | \
                  (anchor_B.index.get_level_values(chromosome_name_B) == self.y_chromosome)
            anchor_A = anchor_A.subset(sel)
            anchor_B = anchor_B.subset(sel)
        self.points(anchor_A.center, anchor_B.center, where, color, size, marker, alpha)

    def _filter_stripe_list(self, stripes):
        return [
            s for s in stripes
            if (s.chromosome == self.x_chromosome or s.chromosome == self.y_chromosome)
               and (
                       (s.anchor_start <= self.xlim.end and s.anchor_end >= self.xlim.start) or
                       (s.anchor_start <= self.ylim.end and s.anchor_end >= self.ylim.start)
               )
        ] if self.x_chromosome is not None else list(stripes)

    def stripe_loop_clusters(
            self, stripes, where='both', cmap='redgreen', color='stripe_score', linewidth=1.0, alpha=None, clim=(0, 1)
    ):
        assert where not in ('a', 'asis')
        stripes = self._filter_stripe_list(stripes)
        if len(stripes) == 0:
            return
        if isinstance(cmap, str):
            if cmap == 'redgreen':
                cmap = LinearSegmentedColormap.from_list('redgreen', [(1, 0, 0), (0, 1, 0)])
            else:
                cmap = plt.cm.get_cmap(cmap)
        rects_df = DataFrame.from_records([
            (
                s.chromosome,
                lc.left_start if s.direction == 'L' else min(lc.left_start, s.position),
                lc.left_end if s.direction == 'L' else max(lc.left_end, s.position),
                s.chromosome,
                lc.right_start if s.direction == 'R' else min(lc.right_start, s.position),
                lc.right_end if s.direction == 'R' else max(lc.right_end, s.position),
                s.features[color][i]
            )
            for s in stripes
            for i, lc in enumerate(s.loop_clusters)
        ], columns=['chromosome_x', 'start_x', 'end_x', 'chromosome_y', 'start_y', 'end_y', 'score'])
        rects_df = rects_df.set_index(['chromosome_x', 'chromosome_y'], append=True)
        rx = RangeSeries(rects_df, suffix='_x')
        ry = RangeSeries(rects_df, suffix='_y')
        color_vals = cmap((rects_df.score - clim[0]) / (clim[1] - clim[0]))
        self.regions2d(
            rx, ry, where,
            color=color_vals, linewidth=linewidth, fill=False, alpha=alpha,
            chromosome_name_A='chromosome_x', chromosome_name_B='chromosome_y'
        )

    def stripe_details(
            self, stripes, where='upper', cmap='redgreen', linewidth=1.0, alpha=None, color='stripe_score', clim=(0, 1)
    ):
        assert where not in ('a', 'asis')
        stripes = self._filter_stripe_list(stripes)
        if isinstance(cmap, str):
            if cmap == 'redgreen':
                cmap = LinearSegmentedColormap.from_list('redgreen', [(1, 0, 0), (0, 1, 0)])
            else:
                cmap = plt.cm.get_cmap(cmap)
        self.stripe_loop_clusters(
            stripes, where=where, color=color, cmap=cmap, clim=clim,
            linewidth=linewidth, alpha=alpha
        )
        for stripe in stripes:
            sf = stripe.features
            pos = stripe.position
            prev = pos
            for i, lc in enumerate(stripe.loop_clusters):
                if stripe.direction == 'R':
                    coords = prev, pos, lc.leafs_start, pos
                    prev = lc.leafs_end
                else:
                    coords = pos, prev, pos, lc.leafs_end
                    prev = lc.leafs_start
                if where == 'lower':
                    coords = coords[1], coords[0], coords[3], coords[2]
                color_val = cmap((sf[color][i] - clim[0]) / (clim[1] - clim[0]))
                self.line(*coords, linewidth=linewidth, color=color_val, alpha=alpha)
            txt = f"{stripe.id}({sf['stripe_score'][-1]:.2f}|{sf['worst_gap_score'].mean():.2f}|{sf['cross_score'].mean():.2f}|{sf['orientation_score'].max():.1f})"
            self.text(stripe.leafs_end, pos, txt,
                      ha='left' if stripe.direction == 'R' else 'right', va='center',
                      fontsize='x-small', fontweight='normal')

    def stripes(
            self, stripes, where='both', pad_w=0, pad_l=0,
            color='black', linewidth=1.0, fill=False, alpha=None
    ):
        if self.x_chromosome is not None:
            _chr = stripes.chromosome
            stripes = stripes.loc[(_chr == self.x_chromosome) | (_chr == self.y_chromosome), :]
        assert where not in ('a', 'asis')
        rects_df = DataFrame({
            'chromosome_x': stripes.chromosome,
            'start_x': stripes.anchor_start - pad_w,
            'end_x': stripes.anchor_end + pad_w,
            'chromosome_y': stripes.chromosome,
            'start_y': stripes.start - np.where(stripes.direction == 'L', pad_l, 0),
            'end_y': stripes.end + np.where(stripes.direction == 'R', pad_l, 0)
        }).set_index(['chromosome_x', 'chromosome_y'], append=True)
        rx = RangeSeries(rects_df, suffix='_x')
        ry = RangeSeries(rects_df, suffix='_y')
        self.regions2d(
            rx, ry, where,
            color=color, linewidth=linewidth, fill=fill, alpha=alpha,
            chromosome_name_A='chromosome_x', chromosome_name_B='chromosome_y'
        )

    def axes(self, padding=None, unit: Union[str, int, None] = None, tick_fontsize=8):
        x0 = self.xlim.start
        y0 = self.ylim.start
        r = self.resolution

        if unit is None:
            reg_size = max(self.xlim.end - self.xlim.start, self.ylim.end - self.ylim.start)
            if reg_size < 100_000:
                unit = 'bp'
            elif reg_size < 10_000_000:
                unit = 'kb'
            else:
                unit = 'mb'
        if isinstance(unit, str):
            unit = {
                'bp': 1,
                'kb': 1_000,
                'mb': 1_000_000
            }[unit.lower()]

        def x_fmt(x, pos):
            return f'{math.floor((x0 + x * r) / unit):,d}'

        def y_fmt(y, pos):
            return f'{math.floor((y0 + y * r) / unit):,d}'

        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))        
        self.ax.tick_params(axis='x', labelsize=tick_fontsize)        
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
        self.ax.tick_params(axis='y', labelsize=tick_fontsize)

        self.ax.xaxis.set_label_position('top')

        if padding is not None:
            if hasattr(padding, '__len__'):
                xpad, ypad = padding
            else:
                xpad = ypad = padding
            self.ax.set_xlim(xmin=-xpad, xmax=self.matrix_data.shape[0] + xpad)
            self.ax.set_ylim(ymin=self.matrix_data.shape[1] + ypad, ymax=-ypad)


class HiCPlotter(object):
    def __init__(self, hic_file_paths):
        self._hic_file_paths = hic_file_paths
        self._hic_files = {}
        self._heatmaps = {}

    def get_heatmap(self, idx, region) -> HiCHeatmap:
        path = self._hic_file_paths.loc[idx]
        if isinstance(path, Series):
            assert len(path) == 1
            path = path.iat[0]
        if (path, region) in self._heatmaps:
            return self._heatmaps[(path, region)]
        if path in self._hic_files:
            hic_file = self._hic_files[path]
        else:
            hic_file = hicstraw.HiCFile(path)
            self._hic_files[path] = hic_file
        hm = HiCHeatmap(hic_file, region)
        self._heatmaps[(path, region)] = hm
        return hm

    def heatmap(self, idx, region, max_color=1.0, ax=None):
        hm = self.get_heatmap(idx[:-1], region)
        if ax is None:
            plt.figure(figsize=(6, 6))
        hm.heatmap(max_color=max_color, ax=ax)
        return hm

    def region_details(
            self, idx, region,
            loops=None, stripes=None,
            color='stripe_score', max_color=0.99, ax=None, linewidth=1.5, clim=(0, 1)
    ):
        hm = self.heatmap(idx, region, max_color, ax)

        if loops is not None:
            loops = loops.loc[idx]
            hm.loops(loops, where='b', color='blue', size=5.0, alpha=1.0)

        if stripes is not None:
            stripes = stripes.loc[idx[:-1]]
            stripes_l = [s for s in stripes if s.direction == 'L']
            stripes_r = [s for s in stripes if s.direction == 'R']
            hm.stripe_details(stripes_r, where='upper', color=color, linewidth=linewidth, clim=clim)
            hm.stripe_details(stripes_l, where='lower', color=color, linewidth=linewidth, clim=clim)

        hm.axes()

    def region(
            self, idx, region,
            loops=None, stripes_u=None, stripes_l=None,
            color_u='green', color_l='brown',
            pad_u=0, pad_l=0,
            max_color=0.99, ax=None
    ):
        hm = self.heatmap(idx, region, max_color, ax)

        def _plot_stripes(sdf, where, c, pad):
            if sdf is None:
                return
            elif isinstance(sdf, DataFrame):
                sdf = sdf.loc[idx]
            elif isinstance(sdf, Series):  # series of stripe lists
                sdf = sdf.loc[idx[:-1]]
                assert len(sdf) == 1
                slist = sdf.iat[0]
                sdf = GraphStripeCaller.get_stripes_dataframe(slist).reset_index()
            hm.stripes(sdf, where=where, color=c, pad_w=pad, pad_l=pad)

        _plot_stripes(stripes_l, 'lower', color_l, pad_l)
        _plot_stripes(stripes_u, 'upper', color_u, pad_u)

        if loops is not None:
            loops = loops.loc[idx]
            hm.loops(loops, where='b', color='blue', size=5.0, alpha=1.0)

        hm.axes()

