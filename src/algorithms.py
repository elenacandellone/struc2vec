# -*- coding: utf-8 -*-
"""Random-walk generation on the multilayer context graph.

Python 3 port of the struc2vec++ reference implementation
(github.com/steenfatt/struc2vec).
"""
from collections import deque
import numpy as np
import math, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *


def generate_parameters_random_walk():
    """
    For each layer and node, compute the number of edges incident to the node with weight
    larger than the average weight of all edges in that layer.

    Store as a dictionary (layer -> node -> number of such neighbours)
    """
    logging.info('Loading distances_nets from disk...')

    sum_weights = {}
    amount_edges = {}

    layer = 0
    while is_pickle('distances_nets_weights-layer-' + str(layer)):
        logging.info('Executing layer {}...'.format(layer))
        weights = restore_variable_from_disk('distances_nets_weights-layer-' + str(layer))

        for node, list_weights in weights.items():
            if layer not in sum_weights:
                sum_weights[layer] = 0
            if layer not in amount_edges:
                amount_edges[layer] = 0

            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    average_weight = {}
    for layer in sum_weights.keys():
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]

    logging.info("Saving average_weights on disk...")
    save_variable_on_disk(average_weight, 'average_weight')

    amount_neighbours = {}

    layer = 0
    while is_pickle('distances_nets_weights-layer-' + str(layer)):
        logging.info('Executing layer {}...'.format(layer))
        weights = restore_variable_from_disk('distances_nets_weights-layer-' + str(layer))

        amount_neighbours[layer] = {}

        for node, list_weights in weights.items():
            cont_neighbours = 0
            for w in list_weights:
                if w > average_weight[layer]:
                    cont_neighbours += 1
            amount_neighbours[layer][node] = cont_neighbours

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving amount_neighbours on disk...")
    save_variable_on_disk(amount_neighbours, 'amount_neighbours')


def choose_neighbour(v, graphs, alias_method_j, alias_method_q, layer):
    v_list = graphs[layer][v]

    idx = alias_draw(alias_method_j[layer][v], alias_method_q[layer][v])
    v = v_list[idx]

    return v


def exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours):
    """
    Execute a single random walk starting at a given node.
    """
    original_v = v
    t0 = time()
    initial_layer = 0
    layer = initial_layer

    path = deque()
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        if r < 0.3:
            v = choose_neighbour(v, graphs, alias_method_j, alias_method_q, layer)
            path.append(v)

        else:
            r = random.random()
            if r > prob_moveup(amount_neighbours[layer][v]):
                if layer > initial_layer:
                    layer = layer - 1
            else:
                if (layer + 1) in graphs and v in graphs[layer + 1]:
                    layer = layer + 1

    t1 = time()
    logging.info('RW - vertex {}. Time : {}s'.format(original_v, (t1 - t0)))

    return path


def walk_seed(seed, walk_iter):
    """
    Seed for one walk iteration, derived from the run seed (None if unseeded). Seeding per iteration keeps
    the walks independent of how iterations are scheduled on worker processes, and gives forked workers
    different random streams.
    """
    if seed is None:
        return None
    return int(np.random.SeedSequence([seed, walk_iter]).generate_state(1)[0])


def exec_random_walks_for_chunk(vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours,
                                seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours))
    return walks


def generate_random_walks_large_graphs(num_walks, walk_length, workers, vertices, seed=None):
    logging.info('Loading distances_nets from disk...')

    graphs = restore_variable_from_disk('distances_nets_graphs')
    alias_method_j = restore_variable_from_disk('nets_weights_alias_method_j')
    alias_method_q = restore_variable_from_disk('nets_weights_alias_method_q')
    amount_neighbours = restore_variable_from_disk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time()

    walks = deque()
    vertices = list(vertices)

    rng = random.Random(seed)
    for walk_iter in range(num_walks):
        rng.shuffle(vertices)
        logging.info("Execution iteration {} ...".format(walk_iter))
        walk = exec_random_walks_for_chunk(vertices, graphs, alias_method_j, alias_method_q, walk_length,
                                           amount_neighbours, walk_seed(seed, walk_iter))
        walks.extend(walk)
        logging.info("Iteration {} executed.".format(walk_iter))

    t1 = time()
    logging.info('RWs created. Time : {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)


def generate_random_walks(num_walks, walk_length, workers, vertices, seed=None):
    """
    Execute and save random walks. With a seed, the walks file is identical across runs.
    """
    logging.info('Loading distances_nets on disk...')

    graphs = restore_variable_from_disk('distances_nets_graphs')
    alias_method_j = restore_variable_from_disk('nets_weights_alias_method_j')
    alias_method_q = restore_variable_from_disk('nets_weights_alias_method_q')
    amount_neighbours = restore_variable_from_disk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time()

    walks = deque()
    vertices = list(vertices)

    if workers > num_walks:
        workers = num_walks

    rng = random.Random(seed)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            rng.shuffle(vertices)
            # submit a copy: the executor pickles arguments lazily, so later shuffles of
            # `vertices` could otherwise leak into iterations not yet sent to a worker
            job = executor.submit(exec_random_walks_for_chunk, list(vertices), graphs, alias_method_j,
                                  alias_method_q, walk_length, amount_neighbours, walk_seed(seed, walk_iter))
            futures[job] = walk_iter
        logging.info("Receiving results...")
        walks_by_iter = {}
        for job in as_completed(futures):
            r = futures[job]
            walks_by_iter[r] = job.result()
            logging.info("Iteration {} executed.".format(r))
            del futures[job]
        # concatenate in iteration order so the walks file does not depend on scheduling
        for walk_iter in sorted(walks_by_iter):
            walks.extend(walks_by_iter[walk_iter])

    t1 = time()
    logging.info('RWs created. Time: {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)


def save_random_walks(walks):
    with open('random_walks.txt', 'w') as fh:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v) + ' '
            line += '\n'
            fh.write(line)


def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = (x / (x + 1))
    return p


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
