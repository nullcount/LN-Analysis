# centers, leaves = 1000, 1000
#
# # we only want the top nodes respectively
# from hyperopt import hp, fmin, tpe, STATUS_OK
# import hyperopt.pyll
# from hyperopt.pyll import scope
# from hyperopt.pyll.stochastic import sample
# import numpy as np
#
#
# # Add a new method as you want
# @scope.define
# def subbleset(choices, num_choices):
#     return np.random.choice(choices, size=num_choices, replace=False)
#
#
# choices = range(0, 100)
#
# # Define the space like below and use in fmin
# space = {
#     "centers": scope.subbleset(choices, hp.randint("num_choices", 0, 101)),
#     "leaves": scope.subbleset(choices, hp.randint("_num_choices", 0, 101))
# }
#
#
# # define an objective function
# def objective(args):
#     # indices of the top 100 leaf nodes and center nodes
#     # leaf nodes are those that appear as leafs of center nodes in a bfs
#     # and are endpoints in longest shortest paths
#     # center nodes are nodes that appear most often in shortest paths
#
#     _leaves = args["leaves"]
#     _centers = args["centers"]
#
#     cost = abs(np.sum(_centers) ** np.sum(_leaves))
#     # print(cost)
#     return {'loss': cost, 'status': STATUS_OK}
#
#
# """
# # define a search space
# space = {
#     "leaves":[hp.choice(f'l{i}' ,[0,1]) for i in range(leaves)],
#     "centers":[hp.choice(f'c{i}',[0,1]) for i in range(centers)]
# }
# """
# # minimize the objective over the space
# best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)
#
# print(best)
# # print([round(v) for v in best.values()])
#
# print(hyperopt.space_eval(space, best))

from os.path import join
from os import getcwd
from lnsimulator.ln_utils import preprocess_json_file
from helpers import *
import lnsimulator.simulator.transaction_simulator as ts
from random import randint, choices
from string import ascii_lowercase, digits
from collections import defaultdict
import networkx as nx
from itertools import combinations


# config = getConfig()
#
# data_dir = join(getcwd(), "cleaned_graphs")
# graph_loc = join(data_dir, "1611045003.json")  # add: tmp_json = tmp_json["graph"] after line 11 in ln_utils
# directed_edges = preprocess_json_file(graph_loc)
# providers = get_merchants()
#
# amount = 60000
# count = 7000
# epsilon = 0.8
# drop_disabled = True
# drop_low_cap = True
# with_depletion = True
#
# simulator = ts.TransactionSimulator(directed_edges, providers, amount, count, drop_disabled=drop_disabled,
#                                     drop_low_cap=drop_low_cap, epsilon=epsilon, with_depletion=with_depletion)
#
# cheapest_paths, _, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=False)
# output_dir = "test"
# total_income, total_fee = simulator.export(output_dir)


# NOTE: use rich club coefficient to find groups of well connected nodes.


def generate_node_id():
    length = 66
    id = ''.join(choices(ascii_lowercase + digits, k=length))
    return id


def generate_channel(src, trg):
    # 1 in 271388828513075199 chance of generating non unique id
    # a should be whatever is 1 + max channel id in most current graph
    channel_dict = {
        "snapshot_id": 0,
        "src": src,
        "trg": trg,
        "last_update": 1612660305,
        "channel_id": randint(728611171486924801, 999999999999999999),
        "capacity": 800000,
        "disabled": 0,
        "fee_base_msat": 1000,
        "fee_rate_milli_msat": 1,
        "min_htlc": 1000
    }
    return channel_dict


"""
template info
more detailed info can be found in the file titled '5_number_summary.txt'

snapshot_id, src, trg, last_update, channel_id, capacity, disabled, fee_base_msat, fee_rate_milli_msat, min_htlc

snapshot_id:0,
src:our node id (random string 66 characters),
trg:node id of node we share a channel with,
last_update: some timestamp,
channel_id: unique 18 digit number x | 728611171486924800 < x < 999999999999999999
capacity: 800000
disabled: 0,
fee_base_msat: 1000
fee_rate_milli_msat: 1
min_htlc: 1000


Mean Capacity ~ 3,000,000
50% of values fall between 176,623 and 3,000,000

Mean Base Fee ~ 1320
50% of values fall between 1 and 1,000
Most frequent is 1,000

Mean Fee Rate ~ 216
50% of values fall between 1 and 10
Most frequent is 1

Name: min_htlc, dtype: float64
Mean Min HTLC ~ 848
>50% of values are 1000
"""

"""
Search space:
4 Channels from the union (intersection?) of:
    - nodes belonging to longest shortest paths
    - nodes with high betweeness centrality
    - nodes that have 'low fee routing diversity' 
"""


def get_leaves(tree):
    leaves = []
    for node, degree in dict(tree.out_degree).items():
        if degree == 0:
            leaves.append(node)
    return leaves


def create_search_space():
    # first create a set containing nodes belonging to the longest shortest paths
    pass


def generate_long_paths(G, nodes):
    """
    pick 2 random points, find the shortest path between them (hops)
    make a list of shortest paths and keep a count of the occurences of nodes in each path (dict: node->int)
    """
    node_count = defaultdict(int)
    long_paths = set()

    for i in range(int(len(nodes) * 10)):  # more iterations -> higher chance of finding long shortest paths
        n_1, n_2 = choices(nodes, k=2)
        path = nx.shortest_path(G, source=n_1, target=n_2)
        if len(path) >= 6:
            long_paths.add((path[0], path[-1]))
        for node in path:
            node_count[node] += 1
    return long_paths, node_count


def generate_all_paths(G, nodes):
    """
    generate all pairs of nodes, find the shortest path between them (hops)
    make a list of shortest paths and keep a count of the occurences of nodes in each path (dict: node->int)
    """
    node_count = defaultdict(int)
    long_paths = set()
    all_pairs = combinations(nodes, 2)
    for n_1, n_2 in all_pairs:  # more iterations -> higher chance of finding long shortest paths
        if n_1 == n_2: continue
        path = nx.shortest_path(G, source=n_1, target=n_2)
        if len(path) >= 6:
            long_paths.add((path[0], path[-1]))
        for node in path:
            node_count[node] += 1
    return long_paths, node_count


"""
Objective function:
Considers:
    - % of successful payments routed (maximize)
    - number of -1 cheapest paths (minimize)
    - total income (maximize) 
    - number of transactions processed (maximize)
    - cost to rebalance entire capacity (minimize)
    - average age of connected nodes (maximize avg, minimize std dev)
"""


def main():
    # we want to make channels between nodes with high betweeness centrality to nodes with low closeness centrality

    G = getGraph(getDict("cleaned_graphs/1608751820-graph.json"))
    close_dict = nx.closeness_centrality(G)
    sorted_close_dict = sorted(list(close_dict.items()), key=lambda kv: kv[1], reverse=True)
    most_close = [x[0] for x in sorted_close_dict[:100]]
    least_close = [x[0] for x in sorted_close_dict[-100:]]

    between_dict = nx.betweenness_centrality(G)
    sorted_btwn_dict = sorted(between_dict.items(), key=lambda kv: kv[1], reverse=True)
    most_between = [x[0] for x in sorted_btwn_dict[:100]]
    least_between = [x[0] for x in sorted_btwn_dict[-100:]]

    # just for fun
    close_and_between = list(set.intersection(set(most_close), set(most_between)))
    far_and_beyond = list(set.intersection(set(least_close), set(least_between)))

    long_paths, between_dict = generate_long_paths(G, list(G.nodes()))
    lp, nc = generate_long_paths(G, least_close)
    alp, anc = generate_all_paths(G, least_close)

    print("Number of long paths found by picking random nodes from entire network:")
    print(len(long_paths))
    print("Number of long paths found by picking random pairs of 'unclose' nodes:")
    print(len(lp))
    print("Total number of long paths in the bottom 100 closest nodes:")
    print(len(alp))


main()

"""
Notes:
There are 1528 paths with length >=6
~500 (33%) of those paths are between bottom 100 nodes in terms of closeness centrality.
~10x as many long paths are found when picking random nodes from the bottom 100 vs. the entire graph
"""
