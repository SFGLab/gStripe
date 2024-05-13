from __future__ import annotations

import gc
import logging
import numpy as np

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional, Dict, List, Tuple

from .utils import MessagePrinter, Timer


class SharedNumpyArray(object):
    __slots__ = ['_name', '_shm', '_array', '_shape', '_dtype']
    __state = ('_name', '_shape', '_dtype')

    def __init__(self, name, shape=None, dtype=None):
        super().__init__()
        assert name
        self._shm = None
        self._array = None
        self._name = name
        self._shape = None
        self._dtype = None
        if dtype is not None or shape is not None:
            if not (dtype is not None and shape is not None):
                raise ValueError("Must specify both shape and dtype")
            self._shape = tuple(shape) if hasattr(shape, '__len__') else (int(shape),)
            self._dtype = np.dtype(dtype)

    def get_template(self):
        return SharedNumpyArray(self.name, shape=self.shape, dtype=self.dtype)

    def create(self, data):
        if self._shm is not None:
            raise Exception(f"Shared memory already exists: {self._name}")
        if self._shape is not None and data.shape != self._shape:
            raise ValueError(f"Shape mismatch with previously declared: {data.shape} vs {self._shape}")
        if self._dtype is not None and data.dtype != self._dtype:
            raise ValueError(f"Dtype mismatch with previously declared: {data.dtype} vs {self._dtype}")
        self._shape = data.shape
        self._dtype = data.dtype
        data_size = np.prod(data.shape) * np.dtype(data.dtype).itemsize
        shm = SharedMemory(create=True, name=self._name, size=data_size)
        array = np.ndarray(shape=self._shape, dtype=self._dtype, buffer=shm.buf)
        array[:] = data[:]
        self._shm = shm
        self._array = array
        return self

    def get(self, shape=None, dtype=None):
        if self._shm is None:
            if dtype is not None or shape is not None:
                if not (dtype is not None and shape is not None):
                    raise ValueError("Must specify both shape and dtype")
                if self._shape is not None or self._dtype is not None:
                    raise ValueError("Shape and dtype already specified")
                self._shape = tuple(shape) if hasattr(shape, '__len__') else (int(shape),)
                self._dtype = np.dtype(dtype)
            elif self._shape is None or self._dtype is None:
                raise ValueError("Shape and dtype not previously declared, you must specify them now.")
            shm = SharedMemory(name=self._name)
            array = np.ndarray(self._shape, dtype=self._dtype, buffer=shm.buf)
            self._shm = shm
            self._array = array
        return self

    def release(self):
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
        self._shm = None

    @property
    def name(self):
        return self._name

    @property
    def array(self):
        if self._shm is None:
            raise Exception("Shared memory not initialized")
        return self._array

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __getstate__(self):
        state = tuple(getattr(self, slot) for slot in self.__state)
        print(f'GS {state}')
        return state

    def __setstate__(self, state):
        print(f'SS {state}')
        self._shm = None
        self._array = None
        for slot, value in zip(self.__state, state):
            setattr(self, slot, value)


class SharedResourcesManager(object):
    instances: Dict[SharedResourcesManager] = {}

    def __init__(self, key):
        if not hasattr(self, '_already_inited'):
            self._already_inited = True
            self._key_str = SharedResourcesManager._get_key_str(key)
            self._items = {}

    def __new__(cls, key, *args, **kwargs):
        key_str = SharedResourcesManager._get_key_str(key)
        instances = SharedResourcesManager.get_instances()
        if key_str not in instances:
            instance = super().__new__(cls)
            instances[key_str] = instance
        else:
            instance = instances[key_str]
        return instance

    @staticmethod
    def _get_key_str(key):
        if isinstance(key, str):
            key_str = key
        elif isinstance(key, type):
            key_str = key.__name__
        else:
            key_str = type(key).__name__
        return key_str

    @classmethod
    def get_instances(cls):
        return cls.instances

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value

    def __contains__(self, key):
        return key in self._items

    def __delitem__(self, key):
        del self._items[key]

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def __len__(self):
        return len(self._items)

    def __getnewargs__(self):
        return (self._key_str,)

    def __getstate__(self):
        return (self._key_str,)

    def __setstate__(self, state):
        self._key_str, = state

    def __str__(self):
        return f'<{self.__class__},{id(self)},{self.keys()}>'


@dataclass
class Task(object):
    def __post_init__(self):
        self.init()

    def init(self):
        self._logger = None
        self._printer = MessagePrinter()

    def create_jobs(self, runner: TaskRunner):
        raise NotImplemented()    

    def run_jobs(self, runner: TaskRunner):
        jobs = self.create_jobs(runner)
        results = runner.run_jobs(jobs)
        return results

    @classmethod
    def put_inherited(cls, key, obj, runner: TaskRunner = None):
        srm = SharedResourcesManager(cls)
        srm[key] = obj
        if runner:
            runner.register_shared(obj)

    @classmethod
    def get_inherited(cls, key):
        srm = SharedResourcesManager(cls)
        return srm[key]
    
    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, new_logger: logging.Logger):
        self._logger = new_logger

    @property
    def printer(self):
        return self._printer
    
    @printer.setter
    def printer(self, new_printer: MessagePrinter):
        self._printer = new_printer
    
    def timer(self, name) -> Timer:
        return Timer(name, self.printer)


@dataclass
class MockFuture(object):
    id: int
    function: Callable[[Any], Any]
    args: Tuple[Any]
    kwargs: Dict[Any, Any]
    _result: Any = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def run(self):
        self._result = self.function(*self.args, **self.kwargs)

    def result(self):
        return self._result


class MockExecutor(object):
    def __init__(self, **kwargs):
        self.submitted = None

    def submit(self, fun, *args, **kwargs):
        i = len(self.submitted)
        future = MockFuture(i, fun, args, kwargs)
        self.submitted.append(future)
        return future

    def __enter__(self):
        self.submitted = []
        return self

    def __exit__(self, type, value, traceback):
        return False


def mock_as_completed(futures, timeout=None):
    for future in futures:
        future.run()
        yield future


@dataclass
class TaskRunner(object):
    max_workers: int = 0  # 'Maximum number of workers (default: auto)'
    force_multiprocessing: bool = False  # 'Force multiprocessing even if max_workres == 1'
    _managed_objects: List[SharedNumpyArray] = None
    _logger: Optional[logging.Logger] = None

    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, new_logger: logging.Logger):
        self._logger = new_logger

    @classmethod
    def invoke_run(cls, job):
        result = job.function(*job.args, **job.kwargs)
        return result

    def run_jobs(
            self,
            jobs: List,
            executor_factory = ProcessPoolExecutor,
            executor_extra_kwargs: Dict = None,
            raise_exceptions: bool = False,
            printer: Optional[MessagePrinter] = None
        ):
        if self.max_workers == 1 and not self.force_multiprocessing:
            executor_factory = MockExecutor
            as_completed_fun = mock_as_completed
        else:
            as_completed_fun = as_completed
        gc.collect()
        if printer is None:
            printer = MessagePrinter(None)
        results = self._run(jobs, executor_factory, executor_extra_kwargs, as_completed_fun, raise_exceptions, printer)
        return results

    def _run(self, jobs, executor_factory, executor_extra_kwargs, as_completed_fun, raise_exceptions, printer):
        executor_kwargs = {} if executor_extra_kwargs is None else dict(executor_extra_kwargs)
        executor_kwargs['max_workers'] = self.max_workers if self.max_workers > 0 else None
        results = {}

        with Timer('TaskRunner', printer) as t:            
            with executor_factory(**executor_kwargs) as executor:
                future_to_job = {self.submit_job(job, executor): job for job in jobs}
                n_jobs = len(future_to_job)
                for i, future in enumerate(as_completed_fun(future_to_job.keys()), 1):
                    completed_job = future_to_job[future]
                    try:
                        result = future.result()
                        message = 'OK'
                        if isinstance(result, tuple):
                            result, message = result
                        results[completed_job.name] = result
                    except Exception as exc:
                        t.message(f'Failed {completed_job.name} [{i}/{n_jobs}]: {exc}')
                        if raise_exceptions:
                            raise
                    else:
                        t.message(f'Finished {completed_job.name} [{i}/{n_jobs}]: {message}')
        return results

    @classmethod
    def submit_job(cls, job, executor):
        future = executor.submit(cls.invoke_run, job)
        return future

    def open_managed_context(self):
        self._managed_objects = []

    def close_managed_context(self):
        for obj in self._managed_objects:
            obj.release()

    def __enter__(self):
        self.open_managed_context()
        return self

    def __exit__(self, type, value, traceback):
        self.close_managed_context()
        return False

    def register_shared(self, value: SharedNumpyArray):
        self._managed_objects.append(value)


@dataclass
class Job(object):  # TODO: is this used?
    name: str
    function: Callable[[Any], Any]
    args: Tuple[Any]
    kwargs: Dict[Any, Any]

    def __init__(self, name, function, *args, **kwargs):
        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs
