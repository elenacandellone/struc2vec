# -*- coding: utf-8 -*-
"""Disk / pickle helpers.

Python 3 port of the struc2vec++ reference implementation
(github.com/steenfatt/struc2vec, "Learning Structural Node Representations on
Directed Graphs", Steenfatt et al., COMPLEX NETWORKS 2018).
"""
from time import time
import logging, inspect
import pickle
from itertools import islice
import os
import os.path

dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Pickle cache location. Fixed by default, overridable per run via
# STRUC2VEC_PICKLES so concurrent / back-to-back runs on different graphs can't
# read each other's stale cached pickles.
folder_pickles = os.environ.get('STRUC2VEC_PICKLES', dir_f + '/../pickles/')
if not folder_pickles.endswith('/'):
    folder_pickles += '/'
os.makedirs(folder_pickles, exist_ok=True)


def return_path_struc2vec():
    return dir_f


def is_pickle(fname):
    return os.path.isfile(folder_pickles + fname + '.pickle')


def chunks(data, size=10000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def partition(lst, n):
    lst = list(lst)
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def restore_variable_from_disk(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1 - t0) / 60))

    return val


def save_variable_on_disk(f, name):
    logging.info('Saving variable on disk...')
    t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1 - t0) / 60))

    return
