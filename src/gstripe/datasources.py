from __future__ import annotations

import itertools
import logging
import os
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Tuple, Union, Hashable, List, Optional, Iterable, Callable, Any, Set, Type, Dict

import numpy as np
import pandas as pd


@dataclass
class _DataSource(object):
    name: Hashable
    root: str
    pattern: re.Pattern
    keys: List[str]
    filters: List[Callable[..., bool]]
    parsers: List[Callable[[str], Any]]
    defaults: List[Any]
    key_tuple_type: Type
    fix: Callable[[Tuple], Tuple]
    action: Callable[[str, Tuple], Any]

    def __post_init__(self):
        self.fix = self._fix if self.fix is None else self.fix
        self.action = self._action if self.action is None else self.action

    def get_paths(self, visited_files: Optional[Set] = None, visited_keys: Optional[Set] = None):
        assert os.path.exists(self.root)  # check if still here...
        for current_dir, _, files in os.walk(self.root):
            for filename in files:
                path = os.path.join(current_dir, filename)
                keys = self.get_values(path)
                abs_path = os.path.abspath(path)
                if keys is not None:
                    if visited_files is not None:
                        if abs_path in visited_files:
                            logging.warning('Path already visited: "%s", keys: %s', abs_path, keys)
                        visited_files.add(abs_path)
                    if visited_keys is not None:
                        if keys in visited_keys:
                            logging.warning('Key already visited: %s', keys)
                    transformed = self.action(path, keys)
                    yield transformed, keys

    def get_values(self, path: str):
        m = self.pattern.match(path)
        if m is None:
            return None
        group_dict = m.groupdict()
        keys = list(self.defaults)
        for i, key in enumerate(self.keys):
            if key in group_dict:
                val = self.parsers[i](group_dict[key])
                keys[i] = val
        keys = self.key_tuple_type(*keys)
        keys = self.fix(keys)
        if keys is None:
            return None
        for flt in self.filters:
            if not flt(path, keys):
                return None
        return keys

    def _fix(self, values):
        return values

    def _action(self, path, keys):
        return path


class DataSources(object):
    def __init__(self, root: str, keys: List[str]):
        if not os.path.exists(root):
            raise FileNotFoundError(root)
        self.root = root
        self.sources: Dict[Hashable, List[_DataSource]] = defaultdict(list)
        self.keys = list(keys)
        self.key_tuple_type = namedtuple(self.make_key_tuple_type_name(keys), keys)

    def make_key_tuple_type_name(self, keys):
        return f'DataSources_{hash(self)}_key_tuple'

    def add(
            self,
            name: Hashable,
            pattern: Union[str, re.Pattern],
            dir: Optional[str] = None,
            parsers: Optional[Dict[str, Callable[[str], Any]]] = None,
            filters: Optional[Dict[str, List[Callable[..., bool]]]] = None,
            defaults: Optional[Dict[str, Any]] = None,
            default_parser: Callable[[str], Any] = str,
            fix: Optional[Callable[[Tuple], Tuple]] = None,
            action: Optional[Callable[[str, Tuple], Any]] = None,
            ignore_path_prefix: bool = True,
            alt_root: Optional[str] = None
    ):
        assert name is not None
        if filters is None:
            filters = []
        root = self.root if alt_root is None else alt_root
        if dir is not None:
            root = os.path.join(root, dir)
        if ignore_path_prefix:
            assert isinstance(pattern, str), "Incompatible options"
            pattern = r'^(?:.*\/)?' + pattern
        compiled_pattern = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        defaults_list = [None] * len(self.keys)
        if defaults is not None:
            for key, val in defaults.items():
                defaults_list[self.keys.index(key)] = val
        parser_list = [default_parser] * len(self.keys)
        if parsers is not None:
            for key, parser in parsers.items():
                if key in self.keys:
                    parser_list[self.keys.index(key)] = parser
        source = _DataSource(
            name, root, compiled_pattern, self.keys, filters, parser_list,
            defaults_list, self.key_tuple_type, fix, action
        )
        self.sources[name].append(source)
        return self

    def get_paths(self, name: Hashable, limit_per_source=None, check_unique=True) -> Iterable[Any]:
        assert name in self.sources
        visited_files = set() if check_unique else None
        visited_keys = set() if check_unique else None
        return itertools.chain.from_iterable(
            itertools.islice(
                source.get_paths(visited_files=visited_files, visited_keys=visited_keys),
                limit_per_source
            )
            for source in self.sources[name]
        )

    def get_paths_as_series(self, name: Hashable, limit_per_source=None, check_unique=True, dtype=None) -> pd.Series:
        paths = list(self.get_paths(name, limit_per_source=limit_per_source, check_unique=check_unique))
        if len(paths) == 0:
            _empty = [[] for _ in self.keys]
            if dtype is None:
                dtype = str
            return pd.Series(
                [], dtype=dtype,
                index=pd.MultiIndex(levels=_empty, codes=_empty, names=self.keys)
            )
        index = pd.MultiIndex.from_tuples((keys for _, keys in paths), names=self.keys)
        series = pd.Series([path for path, _ in paths], index=index, name=name)
        if dtype is not None:
            series = series.astype(dtype)
        return series

    def get_paths_as_dataframe(self, limit_per_source=None, check_unique=True, dtypes=None) -> pd.DataFrame:
        if dtypes is None:
            dtypes = [None] * len(self.sources)
        df = pd.DataFrame({
            name: self.get_paths_as_series(name, limit_per_source=limit_per_source, check_unique=check_unique, dtype=dt)
            for name, dt in zip(self.sources.keys(), dtypes)
        })
        if len(df) == 0:
            df = pd.DataFrame(columns=self.keys + list(self.sources.keys()))
        df = df.fillna(value='')
        df = df.sort_index()
        return df

    @staticmethod
    def read_and_concat_dataframes(
            paths: pd.Series,
            fix: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
            sort: bool = False,
            read_function: Callable[[str, ...], pd.DataFrame] = pd.read_csv,
            ignore_index: bool = False,
            replace_row_idx: Optional[str] = None,
            **read_kwargs
    ) -> pd.DataFrame:
        input_dfs = [
            fix(read_function(path, **read_kwargs))
            for path in paths
        ]
        res_df = pd.concat(input_dfs, keys=paths.index, sort=sort, ignore_index=ignore_index)
        if replace_row_idx is not None:
            res_df = res_df.set_index(pd.RangeIndex(len(res_df), name=replace_row_idx), append=True)
            res_df = res_df.reset_index(-2, drop=True)
        return res_df

    @staticmethod
    def write_to_multiple_files(
            df: pd.DataFrame,
            prefix: str,
            by: Optional[Iterable[str]] = None,
            write_fun: Callable[[pd.DataFrame, str, ...], None] = lambda df, path, **kwargs: df.to_csv(path, **kwargs),
            fix: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
            name_part_sep: str = '_',
            suffix='.csv',
            create_dir=True,
            skip_empty=False,
            **write_fun_kwargs
    ) -> Tuple[str, str, Tuple]:
        sanitize = re.escape

        root, name = os.path.split(prefix)
        if not os.path.exists(root):
            if create_dir:
                os.makedirs(root)
            else:
                raise FileNotFoundError(f'Root dir does not exits: {root}')

        if by is None:  # TODO: inefficient?
            by = df.index
            names = df.index.names
        else:
            by = list(by)
            names = by

        dtypes = None
        unique_values = [set() for _ in names]  # TODO: inefficient?
        for idx, gdf in df.groupby(by, observed=True):
            if dtypes is None:
                dtypes = [type(v) for v in idx]
            for i in range(len(names)):
                unique_values[i].add(idx[i])
            path = prefix + name_part_sep.join(str(part) for part in idx) + suffix
            fixed_df = fix(gdf)
            if fixed_df is not None and (len(fixed_df) > 0 or not skip_empty):
                write_fun(fixed_df, path, **write_fun_kwargs)

        regexes = []
        for dtype, values in zip(dtypes, unique_values):
            if np.issubdtype(dtype, np.integer):
                rx = r'\d+' if all(v >= 0 for v in values) else r'\-?\d+'
            elif np.issubdtype(dtype, np.number):
                rx = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
            else:
                chars_set = set()
                for v in values:
                    chars_set.update(v)
                chars = ''.join(chars_set)
                if chars.isalnum():
                    rx = r'\w+'
                elif len(chars) <= 3:
                    rx = fr'[{sanitize(chars)}]+'
                else:
                    nonalnum = ''.join(c for c in chars if not c.isalnum())
                    rx = fr'[\w{sanitize(nonalnum)}]+'
            regexes.append(rx)

        parts = (fr'(?P<{sanitize(part)}>{rx})' for part, rx in zip(names, regexes))
        restore_regex_template = sanitize(name) + name_part_sep.join(parts) + sanitize(suffix) + '$'
        return root, restore_regex_template, tuple(dtypes)
