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
import pandas as pd
import lnsimulator.simulator.transaction_simulator as ts

data_dir = join(getcwd(), "graphs")
graph_loc = join(data_dir, "1608749883.json")  # add: tmp_json = tmp_json["graph"] after line 11 in ln_utils
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