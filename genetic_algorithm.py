from helpers import *
from random import sample, randrange, uniform, seed
from tqdm import tqdm
import math
from heapq import nlargest
import numpy as np


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
        self.closeness = 0
        self.anticloseness = 0
        self.betweenness = 0
        self.antibetweenness = 0
        self.path_length_sum = 0
        self.coefficients = [0.0, 0.5, 0.0, 0.0, 0.5]  # adds up to 1, indicates which factors we value most

    def set_fitness(self, closeness, anticloseness, betweenness, antibetweenness, path_length_sum):
        self.path_length_sum = path_length_sum
        self.closeness = closeness
        self.anticloseness = anticloseness
        self.betweenness = betweenness
        self.antibetweenness = antibetweenness

        factors = [self.closeness, self.anticloseness, self.betweenness, self.antibetweenness, self.path_length_sum]

        # maximize a vectors distance from zero vector
        self.fitness = 0
        _sum = 0
        for i in range(5):
            _sum += self.coefficients[i] * pow(factors[i], 2)
        self.fitness = round(math.sqrt(_sum), 5)

    def mutate(self):
        """
        Pick a zero index and one index out of z_indices and o_indices respectively and swap them
        """
        # pick random index from each list
        i = randrange(self.z_indices.size)
        j = randrange(self.o_indices.size)

        # get the indices of the bitstring to swap
        z_index = self.z_indices[i]
        o_index = self.o_indices[j]

        # swap the bits in the bit string
        self.bitstring[z_index], self.bitstring[o_index] = self.bitstring[o_index], self.bitstring[z_index]

        # swap the elements between the index lists
        self.z_indices[i], self.o_indices[j] = self.o_indices[j], self.z_indices[i]

    def __repr__(self):
        # return "{} {} {} {}".format(self.closeness, self.anticloseness, self.betweenness, self.antibetweenness)
        return "{}".format(self.fitness)


class GeneticAlgorithm:
    def __init__(self,
                 node_id,
                 base_graph,
                 num_edges,
                 popsize,
                 num_generations,
                 mrate,
                 keep_best,
                 btwn_dict,
                 close_dict,
                 shortest_path_lengths,
                 ):
        # Heuristic Data
        self.close_dict = close_dict
        self.btwn_dict = btwn_dict
        self.short_dict = shortest_path_lengths

        # Search Space
        self.base_graph = base_graph
        self.nodes = filter_nodes(self.base_graph, self.btwn_dict, self.close_dict)
        self.index_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        # self.cluster_dict = get_cluster_dict(base_graph, node_list=self.nodes, num_edges=num_edges)
        self.graph_size = len(self.nodes)

        # GA Params
        self.num_edges = num_edges
        self.popsize = popsize
        self.num_generations = num_generations
        self.mrate = mrate
        self.keep_best = keep_best
        self.population = []
        self.node_id = node_id
        self.best_individuals = None
        self.minimum_length = (self.num_edges * (self.num_edges - 1)) / 2

    def run(self):
        self.populate()
        pbar = tqdm(range(self.num_generations))
        for i in pbar:
            self.repopulate()
            pbar.set_postfix({"Best": self.best_individuals[-1]})
        edges = [self.index_to_node[idx] for idx in self.best_individuals[-1].o_indices]

        return edges

    def populate(self):
        for i in range(self.popsize):
            self.population.extend(self.new_random(self.popsize))

    def repopulate(self):
        self.get_fitnesses()
        self.best_individuals = self.get_best_individuals()
        self.select()
        self.mutate_all()
        self.population.extend(self.best_individuals)

    def get_fitnesses(self):
        for individual in self.population:
            # edges = [(self.node_id, self.index_to_node[idx]) for idx in individual.o_indices]
            nodes = [self.index_to_node[idx] for idx in individual.o_indices]  # could probably be an apply on the array
            closeness_sum = 0
            betweenness_sum = 0
            path_length_sum = 0
            for neighbor in nodes:
                closeness_sum += self.close_dict[neighbor]
                betweenness_sum += self.btwn_dict[neighbor]
            for i in range(self.num_edges):
                u = nodes[i]
                for j in range(i + 1, self.num_edges):
                    path_length_sum += self.short_dict[u][nodes[j]]

            individual.set_fitness(closeness=closeness_sum,
                                   betweenness=betweenness_sum,
                                   anticloseness=self.num_edges - closeness_sum,
                                   antibetweenness=self.num_edges - betweenness_sum,
                                   path_length_sum=(path_length_sum - self.minimum_length) / self.num_edges)

    def select(self):
        new_population = []
        self.population = sorted(self.population, key=lambda x: x.fitness)
        groups = 3
        subpopsize = (self.popsize - self.keep_best) // groups
        remainder = (self.popsize - self.keep_best) % groups

        # elitism
        new_population.extend(self.elitism(subpopsize))
        # roulette
        new_population.extend(self.roulette(subpopsize))
        # new random
        new_population.extend(self.new_random(subpopsize + remainder))

        self.population = new_population

    def crossover(self, x, y):
        bitstring = np.zeros(self.graph_size, dtype=np.int)
        # join the list of one indices and sample from it
        indices = np.union1d(x.o_indices, y.o_indices)
        z_indices = np.arange(self.graph_size, dtype=np.int)
        o_indices = np.random.choice(indices, self.num_edges, replace=True)
        z_indices = np.delete(z_indices, o_indices)
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
            bitstring = np.zeros(self.graph_size, dtype=np.int)
            z_indices = np.arange(self.graph_size, dtype=np.int)
            # o_indices = np.array([], dtype=np.int)
            # indices = np.array([], dtype=np.int)
            # for cluster_num in range(self.num_edges):
            #     node = sample(self.cluster_dict[cluster_num], 1)[0]
            #     idx = self.node_to_index[node]
            #     o_indices = np.append(o_indices, z_indices[idx])
            #     indices = np.append(indices, idx)
            # z_indices = np.delete(z_indices, indices)
            nodes = sample(self.nodes, self.num_edges)
            o_indices = np.array([self.node_to_index[node] for node in nodes])
            z_indices = np.delete(z_indices, o_indices)

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

    def get_best_individuals(self):
        return nlargest(self.keep_best, self.population, key=lambda x: x.fitness)


def main():
    # seed(69420)  # freeze randomness (GA only)
    node_id = "this_us"
    test_graph = "cleaned_graphs/1616346695-graph.json"
    num_edges = 5

    base_graph = getGraph(getDict(test_graph))
    betweenness_centralities = normalize_dicts(save_load_betweenness_centralities(base_graph))
    closeness_centralities = normalize_dicts(save_load_closeness_centralities(base_graph))
    shortest_path_lengths = save_load_shortest_path_lengths(base_graph)

    genetic_algorithm = GeneticAlgorithm(node_id=node_id,
                                         base_graph=base_graph,
                                         num_edges=num_edges,
                                         popsize=10,
                                         num_generations=50000,
                                         mrate=0.1,
                                         keep_best=1,
                                         shortest_path_lengths=shortest_path_lengths,
                                         btwn_dict=betweenness_centralities,
                                         close_dict=closeness_centralities,
                                         )
    edge_ids = genetic_algorithm.run()
    print("Edge recommendations:")
    print(genetic_algorithm.best_individuals[-1])
    for edge_id in edge_ids:
        print(edge_id)
    eval_recommendation(test_graph, edge_ids, node_id)


if __name__ == '__main__':
    main()

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
