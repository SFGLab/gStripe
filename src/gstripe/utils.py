from __future__ import annotations

import argparse
import logging
import functools
import time
import dataclasses
import pandas as pd

from logging import Logger
from enum import Enum
from typing import Optional, Callable, Union, List, Hashable


# In Python 3.11 there is a better way
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
LOG_LEVEL_NAMES = {code: name for name, code in LOG_LEVELS.items()}


@functools.total_ordering
class PrintVerbosity(Enum):
    quiet = logging.NOTSET
    message = logging.INFO
    verbose = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented


class MessagePrinter(object):
    def __init__(
            self,
            print_fun: Optional[Callable] = print,
            logger: Optional[Logger] = None,
            verbosity: Optional[PrintVerbosity] = None,
            quiet: bool = False,
            verbose: bool = False
        ) -> None:
        self._print_fun = print_fun
        self._logger = logger
        if verbosity is None:
            if quiet:
                verbosity = PrintVerbosity.quiet
            elif verbose:
                verbosity = PrintVerbosity.verbose
            else:
                verbosity = PrintVerbosity.message
        self._verbosity = verbosity

    @property
    def verbosity(self):
        return self._verbosity
    
    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    def print(self, msg, verbosity=PrintVerbosity.message):        
        if self._logger is not None:
            log_level = verbosity.value
            self._logger.log(log_level, msg)
        elif verbosity >= self.verbosity and self._print_fun is not None:
            self._print_fun(msg)

    def message(self, msg):
        self.print(msg, PrintVerbosity.message)

    def verbose(self, msg):
        self.print(msg, PrintVerbosity.verbose)

    def warning(self, msg):
        self.print(msg, PrintVerbosity.warning)

    def error(self, msg):
        self.print(msg, PrintVerbosity.error)


class Timer(object):
    def __init__(
            self,
            prefix: str,
            output: Union[Callable, Logger, MessagePrinter] = print,
            print_on_start: str = False,
            time_precision: int = 2,
            verbosity = PrintVerbosity.message
        ):
        self._t0 = None
        self._t1 = None
        self.prefix: str = prefix
        self.time_precision = time_precision
        self._printer: MessagePrinter
        if output is None or callable(output):
            self._printer = MessagePrinter(None)
        elif isinstance(output, MessagePrinter):
            self._printer = output
        elif isinstance(output, Logger):
            self._printer = MessagePrinter(print_fun=None, logger=output)        
        self._print_on_start = print_on_start
        self._already_printed = False
        self.verbosity = verbosity

    def format_message(self, msg: str, timed: bool = True) -> str:
        if timed:
            return f'{self.prefix}({self.elapsed:.{self.time_precision}f}s): {msg}'
        else:
            return f'{self.prefix}: {msg}'

    def print(self, msg: str, verbosity: Optional[int] = None, timed: bool = True) -> None:        
        full_msg = self.format_message(msg, timed)
        verbosity = verbosity if verbosity is not None else self.verbosity
        self._printer.print(full_msg, verbosity)

    def started(self, msg: str = 'Started') -> None:
        self.print(msg, self.verbosity, timed=False)

    def finished(self, msg: str = 'Finished') -> None:
        self.print(msg, self.verbosity, timed=True)
        self._already_printed = True

    def message(self, msg: str, timed: bool = True):
        self.print(msg, PrintVerbosity.message, timed=timed)

    def verbose(self, msg: str, timed: bool = True):
        self.print(msg, PrintVerbosity.verbose, timed=timed)

    def exception(self, exc_value):
        self.print(f'Error: {exc_value}', PrintVerbosity.error, timed=False)

    def start(self):
        if self._print_on_start:
            self.started()
        self._t0 = time.time()
        self._t1 = None

    def stop(self):
        self._t1 = time.time()

    @property
    def elapsed(self) -> float:
        t1 = self._t1
        if t1 is None:
            t1 = time.time()
        return t1 - self._t0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type is None:
            if not self._already_printed:
                self.finished()
        else:
            self.exception(exc_val)


def add_argparse_args(cls, parser: argparse.ArgumentParser, skip=None):
    skip = set(skip) if skip is not None else set()
    types = {
        int: int, 'int': int,
        float: float, 'float': float,
        bool: bool, 'bool': bool
    }
    for fi in dataclasses.fields(cls):
        if fi.name in skip or fi.name.startswith('_'):
            continue
        arg_info = {}
        if fi.type is not None:
            arg_info['type'] = types.get(fi.type, str)
        if fi.default is dataclasses.MISSING:
            parser.add_argument(f'{fi.name}', **arg_info)
        else:
            if arg_info['type'] is bool:
                del arg_info['type']
                arg_info['action'] = 'store_false' if fi.default else 'store_true'
            else:
                arg_info['default'] = fi.default
            parser.add_argument(f'--{fi.name}', **arg_info)


def from_argparse_args(cls, argparse_args, converters=None, **xtra_args):
    if converters is None:
        converters = {}
    params = {}
    for fi in dataclasses.fields(cls):        
        name = fi.name
        if name in xtra_args:
            raw_value = xtra_args[name]
            del xtra_args[name]
        elif hasattr(argparse_args, name):
            raw_value = getattr(argparse_args, name)
        else:
            continue
        if name in converters and isinstance(raw_value, str):  # i.e. value was read from class defaults
            value = converters[name](raw_value)
        else:
            value = raw_value
        params[name] = value    
    instance = cls(**params)
    for name, value in xtra_args.items():
        setattr(instance, name, value)
    return instance


def pandas_merge_threeway(
        left_df: pd.DataFrame,
        mid_df: pd.DataFrame,
        right_df: pd.DataFrame,
        mid_to_left: Union[Hashable, List[Hashable]],
        mid_to_right: Union[Hashable, List[Hashable]],
        left_on: Union[Hashable, List[Hashable]],
        right_on: Union[Hashable, List[Hashable], None] = None,
        inner=True,
        suffixes=('_x', '_m', '_y')
):
    if not isinstance(mid_to_left, list):
        mid_to_left = [mid_to_left]
    if not isinstance(mid_to_right, list):
        mid_to_right = [mid_to_right]
    if not isinstance(left_on, list):
        left_on = [left_on]
    if right_on is None:
        right_on = left_on
    elif not isinstance(right_on, list):
        right_on = [right_on]

    column_sets = [left_df.columns, mid_df.columns, right_df.columns]
    new_left_cols, new_mid_cols, new_right_cols = [
        {
            c: f'{c}{suffixes[i]}' for c in column_sets[i]
            if c in column_sets[(i + 1) % 3] or c in column_sets[(i + 2) % 3]
        }
        for i in range(3)  # left, mid, right
    ]

    def _replace(lst, d):
        return [d.get(s, s) for s in lst]

    left_on = _replace(left_on, new_left_cols)
    right_on = _replace(right_on, new_right_cols)
    mid_to_left = _replace(mid_to_left, new_mid_cols)
    mid_to_right = _replace(mid_to_right, new_mid_cols)

    df = mid_df.rename(columns=new_mid_cols)
    df = pd.merge(
        df, left_df.rename(columns=new_left_cols),
        left_on=mid_to_left, right_on=left_on,
        how='inner' if inner else 'left'
    )
    df = pd.merge(
        df, right_df.rename(columns=new_right_cols),
        left_on=mid_to_right, right_on=right_on,
        how='inner' if inner else 'left'
    )
    return df
