#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""struc2vec++ entry point (Python 3 port).

Reference implementation of "Learning Structural Node Representations on Directed
Graphs", Steenfatt, Nikolentzos, Vazirgiannis & Zhao, COMPLEX NETWORKS 2018
(github.com/steenfatt/struc2vec), ported from Python 2 / gensim 3 to
Python 3 / gensim 4.

vs. struc2vec (Ribeiro et al., KDD 2017):
  --directed   BFS rings are still found on the symmetrised graph, but each ring
               node is described by its (in-degree, out-degree) pair; structural
               distance is DTW(in-seq) + DTW(out-seq).
  --weighted   (directed only) in/out degree become weighted strength.
"""
import argparse
import logging
import sys

import graph
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Run struc2vec++.")

    parser.add_argument('--input', metavar='EDGELIST_FILE', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')
    parser.add_argument('--output', metavar='EMB_FILE', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')
    parser.add_argument('--embed-subset', metavar='NODELIST_FILE', type=str, default=None,
                        help='Only compute embeddings for nodes in a given nodelist. '
                             'Still uses the entire graph as context for the embedding.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')
    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')
    parser.add_argument('--bfs-workers', type=int, default=None,
                        help='Number of parallel workers only for BFS stage. Default is using --workers.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false',
                        help='Treat the graph as unweighted (default).')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false',
                        help='Treat the graph as undirected (default).')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool, help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool, help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool, help='optimization 3')
    return parser.parse_args()


def read_graph(args):
    logging.info(" - Loading graph...")
    graph_dict, in_degrees, out_degrees = graph.load_edgelist(args.input, args.directed, args.weighted)
    logging.info(" - Graph loaded.")
    return graph_dict, in_degrees, out_degrees


def read_embedding_set(args):
    vertices = set()
    with open(args.embed_subset, 'r') as f:
        for line in f:
            vertices.add(int(line.strip()))
    return list(vertices)


def learn_embeddings(args):
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                     workers=args.workers, epochs=args.iter)
    model.wv.save_word2vec_format(args.output)
    logging.info("Representations created.")


def exec_struc2vec(args):
    if args.weighted and not args.directed:
        # Edge weights are only defined for the directed variant in the paper.
        sys.stderr.write("WARNING: --weighted is only implemented for --directed graphs; "
                         "falling back to unweighted.\n")
        logging.warning("--weighted ignored (graph is undirected)")
        args.weighted = False

    until_layer = args.until_layer if args.OPT3 else None
    embedding_vertices = read_embedding_set(args) if args.embed_subset else None

    graph_dict, in_degrees, out_degrees = read_graph(args)  # in/out empty if not directed
    G = graph.Graph(graph_dict, args.directed, args.workers, bfs_workers=args.bfs_workers,
                    until_layer=until_layer, in_degrees=in_degrees, out_degrees=out_degrees,
                    embedding_vertices=embedding_vertices)

    if args.OPT1:
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if args.OPT2:
        G.create_vectors()
        G.calc_distances(compact_degree=args.OPT1)
    else:
        G.calc_distances_all_vertices(compact_degree=args.OPT1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()
    G.simulate_walks(args.num_walks, args.walk_length)
    return G


def main(args):
    exec_struc2vec(args)
    learn_embeddings(args)


if __name__ == "__main__":
    main(parse_args())
