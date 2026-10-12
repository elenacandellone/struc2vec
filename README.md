# struc2vec / struc2vec++ (Python 3, fixed directed skeleton)

> **Fork notice.** This is a fork by Elena Candellone of
> [steenfatt/struc2vec](https://github.com/steenfatt/struc2vec), the reference implementation of
> *struc2vec++* (N. Steenfatt, G. Nikolentzos, M. Vazirgiannis, Q. Zhao, "Learning Structural Node
> Representations on Directed Graphs", Complex Networks 2018, doi:10.1007/978-3-030-05414-4_11),
> which is itself a fork of [leoribeiro/struc2vec](https://github.com/leoribeiro/struc2vec).
> Changes relative to steenfatt/struc2vec:
>
> 1. **Python 3 / gensim 4 port** (no algorithmic changes). Install with
>    `pip install -r requirements.txt`. The pickle cache folder can be set per run with the
>    `STRUC2VEC_PICKLES` environment variable.
> 2. **Fixed directed skeleton.** With `--directed`, upstream only added an edge `x y` to the
>    (undirected) BFS skeleton when `x < y`, so every non-reciprocated edge listed with `x > y` was
>    dropped from the neighbourhood rings (in/out degrees were unaffected). On an edgelist with
>    edges in arbitrary orientation this discards about half of the graph. Every edge now joins the
>    skeleton; a repeated directed edge still triggers the duplicates warning.
>
> struc2vec++ options on top of struc2vec:
>
> | flag | effect |
> |------|--------|
> | `--directed` | BFS rings are found on the underlying undirected graph, but every ring node is described by its (in-degree, out-degree) pair; the structural distance at a layer is DTW(sorted in-degrees) + DTW(sorted out-degrees). |
> | `--weighted` | directed only: in/out degrees become weighted strengths (input lines `src dst weight`). On an undirected graph it warns and falls back to unweighted. |
> | `--embed-subset FILE` | only embed the node ids listed in `FILE`, using the whole graph as context. |
> | `--bfs-workers N` | separate number of workers for the BFS stage. |
>
> Example (directed, weighted): `python src/main.py --input graph/tiny_dw.edgelist --output emb/tiny_dw.emb --directed --weighted`

This repository provides a reference implementation of *struc2vec* as described in the paper:<br>
> struc2vec: Learning Node Representations from Structural Identity.<br>
> Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo.<br>
> Knowledge Discovery and Data Mining, SigKDD, 2017.<br>

The *struc2vec* algorithm learns continuous representations for nodes in any graph. *struc2vec* captures structural equivalence between nodes.  

Before to execute *struc2vec*, it is necessary to install the following packages:
<br/>
``pip install futures``
<br/>
``pip install fastdtw``
<br/>
``pip install gensim``

### Basic Usage

#### Example
To run *struc2vec* on Mirrored Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/karate-mirrored.edgelist --output emb/karate-mirrored.emb``

#### Options

To activate optimization 1, use the following option:
``--OPT1 true``
<br/>
To activate optimization 2:
``--OPT2 true``
<br/>
To activate optimization 3:
``--OPT3 true``
<br/>

To run *struc2vec* on Barbell network, using all optimizations, execute the following command from the project home directory:
<br/>
``python src/main.py --input graph/barbell.edgelist --output emb/barbell.emb --num-walks 20 --walk-length 80 --window-size 5 --dimensions 2 --OPT1 True --OPT2 True --OPT3 True --until-layer 6``


You can check out the other options available to use with *struc2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int
		

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *struc2vec*.


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <leo@land.ufrj.br>.

*Note:* This is only a reference implementation of the framework *struc2vec*.
