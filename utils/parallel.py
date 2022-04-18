import logging
from tqdm import tqdm
import joblib
import numpy as np
import multiprocessing as mp
from itertools import chain


class ProgressParallel(joblib.Parallel):
    def __init__(self, total=None, freq=500, **kwargs):
        assert freq > 0
        self._total_tasks = total
        self._prev_reported_tasks = -freq
        self._freq = self._get_optimal_freq(freq, total)
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._need_to_report():
            self._prev_reported_tasks = self.n_completed_tasks
            self._pbar.total = self._total_or_dispatched_tasks()
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    def _need_to_report(self):
        tasks_since_last_update = self.n_completed_tasks - self._prev_reported_tasks
        return ((abs(tasks_since_last_update) >= self._freq) or
                (self.n_completed_tasks == self._total_tasks))

    def _total_or_dispatched_tasks(self):
        return self._total_tasks or self.n_dispatched_tasks

    @staticmethod
    def _get_optimal_freq(requested_freq, total):
        if total:
            return min(requested_freq, total // 10)
        return requested_freq


def _select_parallel_impl(progress, total_tasks, n_jobs=-1,
                          backend='loky', **kwargs):
    """
    backend='multiprocessing' is a bit faster but unstable due to occasional SIGTERM.
    """
    if progress:
        return ProgressParallel(
            total=total_tasks,
            n_jobs=n_jobs,
            backend=backend,
            **kwargs)
    return joblib.Parallel(
        n_jobs=n_jobs,
        backend=backend,
        **kwargs)


class ParallelMap(object):
    def __init__(self, dict_like, progress=True, **kwargs):
        self._dictionary = dict(dict_like)
        self._parallel = _select_parallel_impl(
            progress,
            total_tasks=len(self._dictionary),
            **kwargs)

    def __call__(self, func, *args, **kwargs):
        tasks = (joblib.delayed(func)(k, v, *args, **kwargs)
                 for k, v in self._dictionary.items())
        return dict(self._parallel(tasks))


class ParallelFor(object):
    def __init__(self, list_like, progress=True, **kwargs):
        self._collection = list(list_like)
        self._parallel = _select_parallel_impl(
            progress,
            total_tasks=len(self._collection),
            **kwargs)

    def __call__(self, func, *args, **kwargs):
        tasks = (joblib.delayed(func)(x, *args, **kwargs)
                 for x in self._collection)
        return self._parallel(tasks)


class ThreadApply(ParallelFor):
    def __init__(self, data, n_batches=0, progress=True, out=None, **kwargs):
        if 'backend' in kwargs:
            logging.warning('ThreadApply warning: backend is overwritten as threading')
        kwargs['backend'] = 'threading'

        if n_batches <= 0:
            n_batches = 10 * mp.cpu_count()

        self._data = np.array_split(data, n_batches)
        self._out = out if out else type(self._data)
        super().__init__(self._data, progress, **kwargs)

    def __call__(self, func, *args, **kwargs):
        def wrapper(func):
            def vectorizer(data, *args, **kwargs):
                return [func(value, *args, **kwargs) for value in data]
            return vectorizer
        return self._out(chain(*super().__call__(wrapper(func), *args, **kwargs)))


# Aliases
Map = ParallelMap
For = ParallelFor
