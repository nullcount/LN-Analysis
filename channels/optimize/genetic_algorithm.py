from helpers import *
from random import sample, randrange, uniform, seed
from tqdm import tqdm
import math


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
        self.coefficients = [0, 1, 0, 1]

    def set_fitness(self, closeness, anticloseness, betweenness, antibetweenness):
        self.closeness = closeness
        self.anticloseness = anticloseness
        self.betweenness = betweenness
        self.antibetweenness = antibetweenness

        factors = [self.closeness, self.anticloseness, self.betweenness, self.antibetweenness]

        # maximize a vectors distance from zero vector
        self.fitness = 0
        _sum = 0
        for i in range(4):
            _sum += self.coefficients[i] * pow(factors[i], 2)
        self.fitness = round(math.sqrt(_sum), 5)

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
        return "{} {} {} {}".format(self.closeness, self.anticloseness, self.betweenness, self.antibetweenness)


class GeneticAlgorithm:
    def __init__(self, node_id, base_graph, num_edges, popsize, num_generations, mrate, btwn_dict, close_dict):
        # Heuristic Data
        self.close_dict = close_dict
        self.btwn_dict = btwn_dict

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
            individual.set_fitness(closeness=closeness_sum,
                                   betweenness=betweenness_sum,
                                   anticloseness=self.num_edges - closeness_sum,
                                   antibetweenness=self.num_edges - betweenness_sum)

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

        # capacity
        min_capacity = 2 * 10 ** 6
        tolerance = 1

        # channels
        min_channels = 4
        max_channels = float("inf")
        nodes = []
        # TODO: add last_updated constraint

        # betweenness/closeness
        left, right = 0.60, 0.90  # defines a spread of nodes to choose from
        min_percentile = 0.20
        max_percentile = 1.00
        #  0% |xx|----|xxx|-| 100%
        # only dashed sections are considered 

        for node in self.base_graph.nodes():
            nbrs = list(self.base_graph.neighbors(node))
            if min_channels > len(nbrs) or len(nbrs) > max_channels:
                # must have more channels than this
                continue
            if (left < self.btwn_dict[node] < right) or (left < self.close_dict[node] < right):
                # excludes nodes within this spread
                continue
            if self.btwn_dict[node] < min_percentile and self.close_dict[node] < min_percentile:
                # must have a percentile greater than this
                continue
            if self.btwn_dict[node] > max_percentile and self.close_dict[node] > max_percentile:
                # must have a percentile less than this
                continue
            capacity = sum([int(self.base_graph[node][nbrs[i]]["capacity"]) for i in range(len(nbrs))])
            if capacity < min_capacity - tolerance:
                # must have a capacity greater than this
                continue
            nodes.append(node)
        print(len(nodes))

        return nodes


def main():
    # seed(69420)  # freeze randomness (GA only)
    node_id = "02a32ec469c2dce4937148ac1f6af67aa0054723b7c67a5f95a3c32f00a3caca2e"
    test_graph = "cleaned_graphs/graph.json"
    base_graph = getGraph(getDict(test_graph))
    betweenness_centralities = normalize_dicts(save_load_betweenness_centralities(base_graph))
    closeness_centralities = normalize_dicts(save_load_closeness_centralities(base_graph))

    genetic_algorithm = GeneticAlgorithm(node_id=node_id,
                                         base_graph=base_graph,
                                         num_edges=4,
                                         popsize=100,
                                         num_generations=5000,
                                         mrate=0.1,
                                         btwn_dict=betweenness_centralities,
                                         close_dict=closeness_centralities
                                         )
    edge_ids = genetic_algorithm.run()
    print("Edge recommendations:")
    print(genetic_algorithm.best_individual)
    for edge_id in edge_ids:
        print(edge_id)
    eval_recommendation(test_graph, edge_ids, node_id)


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
