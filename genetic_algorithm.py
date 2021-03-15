from helpers import *
from random import sample, randrange, uniform
from tqdm import tqdm
import math
from scipy.stats import percentileofscore
import lnsimulator.simulator.transaction_simulator as ts
from lnsimulator.ln_utils import preprocess_json_file


class Individual:
    def __init__(self, bitstring, o_indices, z_indices):
        """
        bitstring: encoded string 1 indicates a channel with the node at this index 0 otherwise
        o_indices: list of indices in the bitstring which are equal to 1
        z_indices: list of indices in the bitstring which are equal to 1
        """
        self.bitstring = bitstring
        self.o_indices = o_indices
        self.z_indices = z_indices
        self.fitness = 0
        self.closeness_sum = 0
        self.betweenness_sum = 0

    def set_fitness(self, closeness_sum, betweenness_sum):
        self.closeness_sum = round(closeness_sum, 5)
        self.betweenness_sum = round(betweenness_sum, 5)

        # maximize a vectors distance from zero vector
        self.fitness = math.sqrt(pow(closeness_sum, 2) + pow(betweenness_sum, 2))

    def mutate(self):
        """
        Pick a zero index and one index out of z_indices and o_indices respectively and swap them
        """
        i = randrange(len(self.z_indices))
        self.z_indices[i], self.z_indices[-1] = self.z_indices[-1], self.z_indices[i]
        z_index = self.z_indices.pop()

        j = randrange(len(self.o_indices))
        self.o_indices[j], self.o_indices[-1] = self.o_indices[-1], self.o_indices[j]
        o_index = self.o_indices.pop()

        self.bitstring[z_index], self.bitstring[o_index] = self.bitstring[o_index], self.bitstring[z_index]
        self.o_indices.append(z_index)
        self.z_indices.append(o_index)

    def __repr__(self):
        return "{} {}".format(self.closeness_sum, self.betweenness_sum)


class GeneticAlgorithm:
    def __init__(self, node_id, base_graph, num_edges, popsize, num_generations, mrate, btwn_dict, close_dict):
        # Heuristic Data
        self.close_dict = close_dict
        self.btwn_dict = btwn_dict
        self.normalize_dicts()

        # Search Space
        self.base_graph = base_graph
        self.nodes = self.filter_nodes()
        self.graph_size = len(self.nodes)
        self.index_to_node = {i: node for i, node in enumerate(self.nodes)}

        # GA Params
        self.num_edges = num_edges
        self.popsize = popsize
        self.num_generations = num_generations
        self.mrate = mrate
        self.population = []
        self.node_id = node_id
        self.best_individual = None

    def run(self):
        self.populate()
        pbar = tqdm(range(self.num_generations))
        for i in pbar:
            self.repopulate()
            pbar.set_postfix({"Best": self.best_individual})
        edges = [self.index_to_node[idx] for idx in self.best_individual.o_indices]

        return edges

    def populate(self):
        for i in range(self.popsize):
            self.population.extend(self.new_random(self.popsize))

    def repopulate(self):
        self.get_fitnesses()
        self.best_individual = self.get_best_individual()
        self.select()
        self.mutate_all()
        self.population.append(self.best_individual)

    def get_fitnesses(self):
        for individual in self.population:
            edges = [(self.node_id, self.index_to_node[idx]) for idx in individual.o_indices]
            closeness_sum = 0
            betweenness_sum = 0
            for _, neighbor in edges:
                closeness_sum += self.close_dict[neighbor]
                betweenness_sum += self.btwn_dict[neighbor]
            individual.set_fitness(closeness_sum, betweenness_sum)

    def select(self):
        new_population = []
        self.population = sorted(self.population, key=lambda x: x.fitness)
        groups = 3
        subpopsize = self.popsize // groups
        remainder = self.popsize % groups

        # elitism
        new_population.extend(self.elitism(subpopsize))
        # roulette
        new_population.extend(self.roulette(subpopsize))
        # new random
        new_population.extend(self.new_random(subpopsize + remainder - 1))

        self.population = new_population

    def crossover(self, x, y):
        bitstring = [0 for i in range(self.graph_size)]
        # join the list of one indices and sample from it
        indices = list(set.union(set(x.o_indices), set(y.o_indices)))
        z_indices = list(range(self.graph_size))
        o_indices = sorted(sample(indices, self.num_edges), reverse=True)
        for o_index in o_indices:
            z_indices[o_index], z_indices[-1] = z_indices[-1], z_indices[o_index]
            z_indices.pop()
        for index in o_indices:
            bitstring[index] = 1
        return Individual(bitstring, o_indices, z_indices)

    def elitism(self, n):
        # generate offspring using the elitism strategy
        subpopulation = []
        parents = self.population[:-n]  # lets keep it simple with the elite strategy
        for i in range(n):
            p1, p2 = sample(parents, 2)
            subpopulation.append(self.crossover(p1, p2))
        return subpopulation

    def new_random(self, n):
        # randomly generate new offspring
        subpopulation = []
        for i in range(n):
            bitstring = [0 for _ in range(self.graph_size)]
            z_indices = list(range(self.graph_size))
            o_indices = []
            for _ in range(self.num_edges):
                idx = randrange(len(z_indices))
                z_indices[idx], z_indices[-1] = z_indices[-1], z_indices[idx]
                o_indices.append(z_indices.pop())
            for index in o_indices:
                bitstring[index] = 1
            subpopulation.append(Individual(bitstring, o_indices, z_indices))
        return subpopulation

    def roulette(self, n):
        max = sum([x.fitness for x in self.population])
        subpopulation = []
        for _ in range(n):
            parents = []
            for _ in range(2):
                pick = uniform(0, max)
                current = 0
                for individual in self.population:
                    current += individual.fitness
                    if current > pick:
                        parents.append(individual)
                        break
            subpopulation.append(self.crossover(*parents))
        return subpopulation

    def mutate_all(self):
        for individual in self.population:
            chance = uniform(0, 1)
            if chance >= self.mrate:
                individual.mutate()

    def get_best_individual(self):
        return max(self.population, key=lambda x: x.fitness)

    def filter_nodes(self):
        # returns a list of nodes after constraints
        # possible constraints: capacity, whether we are already connected to them
        min_capacity = 10 ** 6
        tolerance = 1
        nodes = []
        for node in self.base_graph.nodes():
            nbrs = list(self.base_graph.neighbors(node))
            if len(nbrs) < 4:
                continue
            if self.btwn_dict[node] < 0.8 and self.close_dict[node] < 0.8:
                continue
            capacity = sum([int(self.base_graph[node][nbrs[i]]["capacity"]) for i in range(len(nbrs))])
            if capacity < min_capacity - tolerance:
                continue
            nodes.append(node)
        return nodes

    def normalize_dicts(self):
        # closeness_dict
        ckeys, cvalues = list(self.close_dict.keys()), list(self.close_dict.values())
        # subtract from 100 because we want to minimize closeness
        percentiles = [(100 - percentileofscore(cvalues, value, "rank")) / 100 for value in cvalues]
        self.close_dict = dict(zip(ckeys, percentiles))

        # betweeness dict
        bkeys, bvalues = list(self.btwn_dict.keys()), list(self.btwn_dict.values())
        # take actual value because we want to maximize betweenness
        percentiles = [percentileofscore(bvalues, value, "rank") / 100 for value in bvalues]
        self.btwn_dict = dict(zip(bkeys, percentiles))


def eval_recommendation(base_graph, edges, node_id):
    # TODO: Add edges to json before processing into directed edges
    directed_edges = preprocess_json_file("cleaned_graphs/1614938401-graph.json")
    merchant_data = get_merchant_data()
    merchants = list(merchant_data.keys())
    amount = 60000
    count = 7000
    epsilon = 0.8
    drop_disabled = True
    drop_low_cap = False
    with_depletion = True

    simulator = ts.TransactionSimulator(directed_edges,
                                        merchants,
                                        amount,
                                        count,
                                        drop_disabled=drop_disabled,
                                        drop_low_cap=drop_low_cap,
                                        epsilon=epsilon,
                                        with_depletion=with_depletion
                                        )
    cheapest_paths, _, all_router_fees, _ = simulator.simulate(weight="total_fee", with_node_removals=False)

    node_stats = all_router_fees.groupby("node")["fee"].sum()[node_id]


def main():
    node_id = "this_us"
    base_graph = getGraph(getDict("cleaned_graphs/1614938401-graph.json"))
    betweenness_centralities = save_load_betweenness_centralities(base_graph)
    closeness_centralities = save_load_closeness_centralities(base_graph)
    genetic_algorithm = GeneticAlgorithm(node_id=node_id,
                                         base_graph=base_graph,
                                         num_edges=18,
                                         popsize=20,
                                         num_generations=30000,
                                         mrate=0.1,
                                         btwn_dict=betweenness_centralities,
                                         close_dict=closeness_centralities
                                         )
    edges = genetic_algorithm.run()
    print("Edge recommendations:")
    print(genetic_algorithm.best_individual)
    for edge in edges:
        print(edge)
    # eval_recommendation(base_graph, edges, node_id)


if __name__ == '__main__':
    main()

# TODO: evaluate recommendations using LN simulator
# TODO: add parameter to keep x number of best individuals
# TODO: transition to numpy arrays for larger populations
"""
The idea is that we have a single node in the network
we want to add edges between this node and other nodes 
such that we maximize the sum of the betweenness centralities of all our neighbors
and minimize the closeness centrality of all our neighbors.

These objectives are heuristics for what we actually want. We can generate
several candidates with this method quickly (hopefully) and then evaluate them
with a better method, a simulator, which takes more time. 


edge information have a bitstring of length equal to 
the number of nodes we are considering opening channels with
0 = our node does not share a channel with this node
1 = our node does share a channel with this nodes  
"""

"""
Another algorithm of interest: 

find the diameter nodes
make a channel between them
repeat

then add a single channel to the node with highest betweenness
"""
