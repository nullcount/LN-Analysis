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
from helpers import get_merchants
import lnsimulator.simulator.transaction_simulator as ts
from random import randint, choices
from string import ascii_lowercase, digits


data_dir = join(getcwd(), "cleaned_graphs")
graph_loc = join(data_dir, "1611045003.json")  # add: tmp_json = tmp_json["graph"] after line 11 in ln_utils
directed_edges = preprocess_json_file(graph_loc)
providers = get_merchants()

amount = 60000
count = 7000
epsilon = 0.8
drop_disabled = True
drop_low_cap = True
with_depletion = True

simulator = ts.TransactionSimulator(directed_edges, providers, amount, count, drop_disabled=drop_disabled,
                                    drop_low_cap=drop_low_cap, epsilon=epsilon, with_depletion=with_depletion)

cheapest_paths, _, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=False)
output_dir = "test"
total_income, total_fee = simulator.export(output_dir)


# NOTE: use rich club coefficient to find groups of well connected nodes.


def generate_node_id():
    length = 66
    id = ''.join(choices(ascii_lowercase + digits, k=length))
    return id


def generate_channel(src, trg):
    # 1 in 271388828513075199 chance of generating non unique id
    # a should be whatever is 1+ max channel id in most current graph
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


def get_leaves(graph):
    # takes NetworkX Graph as input
    # finds shortest paths between random nodes in the graph
    # gets node with highest betweeness centrality
    # creates a BFS from that node
    # returns the leaves of that tree
    pass


def create_search_space():
    # first create a set containing nodes belonging to the longest shortest paths
    pass


"""
Objective function:
Considers:
    - % change in successful payments routed
    - number of -1 cheapest paths, similar to above (minimize)
    - total income (maximize) 
    - number of transactions processed
    - cost to rebalance entire capacity (minimize)
"""
