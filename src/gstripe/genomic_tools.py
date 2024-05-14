from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd

from typing import Tuple, Union, Iterable, Optional

from .utils import pandas_merge_threeway


_CHROMOSOME_DTYPE = 'str'
_GENOMIC_COORDINATE_DTYPE = 'int64'
_PETCOUNT_DTPYE = 'int64'
_CANDIDATE_PETCOUNT_COLS = ('count', 'numColapsed', 'petcount')

BEDPE_COORD_COLUMNS = [
    (0, 'chromosome_A', _CHROMOSOME_DTYPE),
    (1, 'start_A', _GENOMIC_COORDINATE_DTYPE),
    (2, 'end_A', _GENOMIC_COORDINATE_DTYPE),
    (3, 'chromosome_B', _CHROMOSOME_DTYPE),
    (4, 'start_B', _GENOMIC_COORDINATE_DTYPE),
    (5, 'end_B', _GENOMIC_COORDINATE_DTYPE)
]


def read_raw_contacts_from_bedpe(
        filename, *extra_cols,
        sep: str = '\t',
        skiprows: int = -1,
        comment: str = '#',
        min_petcount: int = 0,
        chromosomes: Union[str, Iterable[str], None] = None,
        intrachromosomal: bool = True,  # TODO: rename: intra inter
        interchromosomal: bool = True,
        nrows: Optional[int] = None,  # TODO: test
        petcount_col: Optional[int] = -1,
        fix_chromosome: bool = True  # TODO: numerical X and Y?
):
    """
    Read interactions from a bedpe file into standardized pandas dataframe. Auto-detects the variant of the file.
    Works with raw file (without header) or outputs from MAPS, HiCCUPS (juicer tools), ChIA-Pipe.
    According to the bedpe convention, the firs six columns will be loaded as coordinates of interacting anchors.
    Chromosome names are read as strings, and might be changed to have the 'chr' prefix (i.e. 'chr13' instead of '13').

    Args:
        path: The path to the bedpe file.
        extra_cols: Description of additional columns to load from the file: each tuple of form `(i, name, dtype)`
                    Indicates, that i-th column from the file is to be added as `name` column with the `dtype`.
        sep: The file column separator.
        skiprows: Number of rows to skip (not including comments) before reading the contents of the file, -1 means auto-detect.
        comment: The character indicating a comment line in the file.
        min_petcount: Do not include interactions with petcount below this value (if the petcount column exists).
        chromosomes: If not `None``, include only interactions with either anchor located on one of the provided chromosomes.
        intrachromosomal: Include interactions with both anchors on the same chromosome.
        interchromosomal: Include interactions with anchors on different chromosomes.
        nrows: Read only the first `nrows` interactions.
        petcount_col: The index of the column contaninig the petcounts. -1 means auto-detect, `None` means do not include petcount.
        fix_chromosome: If true, ensure that the chromosome names have the 'chr' prefix (i.e. 'chr13' instead of '13').

    Returns:
        A pandas dataframe contaning the loaded interactions, with the following columns:
        chromosome_A: str, start_A: int64, end_A: int64, chromosome_B: str, start_B: int64, end_B: int64, [petcount: int64 (if included)],  [any extra columns, with their specified dtypes]
    
    """
    detect_skiprows = skiprows == -1
    first_line = None
    line = 'chr1 0 0'  # sentinel in case of empty file
    # skip comments
    for line in open(filename):        
        if detect_skiprows:
            skiprows += 1
        if first_line is None:
            first_line = line
        if not line.strip().startswith(comment):  
            break        
    # check if line contains a header
    try:
        _, c1, c2, *_ = line.split()
    except ValueError:
        raise ValueError(f'Malformed line {skiprows}: "{line}"')
    try:
        int(c1)
        int(c2)
    except ValueError:
        if detect_skiprows:
            skiprows += 1    
    cols = list(BEDPE_COORD_COLUMNS)
    # detect the petcount column
    if petcount_col == -1:
        petcount_col = None
        if first_line is not None:
            header_cols = first_line.split()
            for col in _CANDIDATE_PETCOUNT_COLS:
                if col in header_cols:
                    petcount_col = header_cols.index(col)
                    break
    if petcount_col is not None:
        cols.append((petcount_col, 'petcount', _PETCOUNT_DTPYE))
    # else: not provided and unable to detect
    usecols = set(i for i, *_ in cols)
    # determine the columns and their dtypes
    for ec in extra_cols:
        i, name, dtype = ec
        if i in usecols:
            raise ValueError(f"Duplicated column index: {i} (column '{name}' of type {dtype})")
        cols.append(ec)
        usecols.add(i)
    # read the file
    df = pd.read_table(
        filename,
        header=None,
        names=[name for _, name, _ in cols],
        dtype={name: dtype for _, name, dtype in cols},
        usecols=usecols,
        na_filter=False,
        sep=sep,
        skiprows=skiprows,
        comment=comment,
        nrows=nrows
    )
    if fix_chromosome:
        chromosome_values = set(df.chromosome_A.unique())
        chromosome_values.update(df.chromosome_B.unique())
        if not all(cv.startswith('chr') for cv in chromosome_values):
            df['chromosome_A'] = 'chr' + df.chromosome_A
            df['chromosome_B'] = 'chr' + df.chromosome_B
    if len(df) == 0:
        df.index = pd.RangeIndex(0)
    selected = pd.Series(True, index=df.index)
    if petcount_col is not None and min_petcount >= 0:
        selected &= df.petcount >= min_petcount
    if not intrachromosomal:
        selected &= df.chromosome_A != df.chromosome_B
    if not interchromosomal:
        selected &= df.chromosome_A == df.chromosome_B
    if isinstance(chromosomes, str):
        selected &= (df.chromosome_A == chromosomes) | (df.chromosome_B == chromosomes)
    elif chromosomes is not None:
        if len(chromosomes) == 1:
            chromosomes = chromosomes[0]
            selected &= (df.chromosome_A == chromosomes) & (df.chromosome_B == chromosomes)
        else:
            selected &= df.chromosome_A.isin(chromosomes) & df.chromosome_B.isin(chromosomes)
    df.index.names = ['contact_idx']
    return df[selected]


def normalize_contacts(bedpe_df: pd.DataFrame, sort_anchors_by='start') -> Tuple[pd.DataFrame, pd.DataFrame]:
    contacts = bedpe_df.copy()
    contacts['length'] = np.abs(contacts.start_A - contacts.start_B + contacts.end_A - contacts.end_B) // 2

    anchor_id_parts = ['chromosome', 'start', 'end']
    contacts_data_cols = ['petcount', 'length']
    id_cols = [
        f'{part}_{which_anchor}'
        for which_anchor in ('A', 'B')
        for part in anchor_id_parts
    ]
    anchors = bedpe_df[id_cols]
    anchors.columns = pd.MultiIndex.from_product([
        ('A', 'B'),
        anchor_id_parts
    ])

    anchors = anchors.stack(0, future_stack=True)
    anchors = anchors.drop_duplicates(subset=anchor_id_parts, keep='first')
    anchors = anchors[anchor_id_parts]  # reorder columns
    anchors['midpoint'] = (anchors.start + anchors.end) // 2
    anchors['length'] = anchors.end - anchors.start + 1
    anchors.sort_values(['chromosome', sort_anchors_by], inplace=True, ignore_index=True)
    anchors.index.names = ['anchor_id']

    anchor_id_a = [f'{s}_A' for s in anchor_id_parts]
    anchor_id_b = [f'{s}_B' for s in anchor_id_parts]

    contacts = pandas_merge_threeway(
        anchors.reset_index(), contacts, anchors.reset_index(),
        anchor_id_a, anchor_id_b, ['chromosome', 'start', 'end'],
        suffixes=('_A', '', '_B')
    )
    contacts = contacts[['anchor_id_A', 'anchor_id_B'] + contacts_data_cols]
    contacts.index.names = ['contact_id']

    return anchors, contacts


@dataclasses.dataclass
class Region:
    chromosome: str
    start: int
    end: int