from helpers import getGraph, getDict, save_load_shortest_path_lengths, save_load_betweenness_centralities
from random import sample, randrange, uniform
from tqdm import tqdm
from math import sqrt


def weighted_random_choice(choices):
    # https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    max = sum(choices.values())
    pick = uniform(0, max)
    current = 0
    for key, value in choices.items():
        current += value
        if current > pick:
            return key


class Individual:
    def __init__(self, bitstring, o_indices, z_indices):
        self.bitstring = bitstring
        self.o_indices = o_indices
        self.z_indices = z_indices
        self.fitness = 0
        self.centrality_sum = 0
        self.lp_sum = 0

    def set_fitness(self, centrality_sum, lp_sum):
        self.centrality_sum = round(centrality_sum, 4)
        self.btwn_sum = round(lp_sum, 4)
        self.fitness = centrality_sum + lp_sum

    def mutate(self):
        # pick one non-zero index and one zero index and swap them
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
        return "{} {}".format(self.centrality_sum, self.btwn_sum)


class GeneticAlgorithm:
    def __init__(self, base_graph, num_edges, popsize, num_generations, mrate, aspl_dict, btwn_dict):
        self.base_graph = base_graph
        self.index_to_node = {i: node for i, node in enumerate(self.base_graph.nodes())}
        self.aspl_dict = aspl_dict
        self.aspl_norm = sum(aspl_dict.values())
        self.btwn_dict = btwn_dict
        self.btwn_norm = sum(btwn_dict.values())
        self.num_edges = num_edges
        self.graph_size = len(self.base_graph)
        self.popsize = popsize
        self.num_generations = num_generations
        self.mrate = mrate
        self.population = []
        self.node_id = "this_us"

    def run(self):
        self.populate()
        pbar = tqdm(range(self.num_generations))
        for i in pbar:
            self.gen_num = i
            self.repopulate()
            pbar.set_postfix({"Best": self.best_individual})

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
            centralities_sum = 0
            lengths_sum = 0
            for _, neighbor in edges:
                centralities_sum += self.btwn_dict[neighbor] / self.btwn_norm
                lengths_sum += self.aspl_dict[neighbor] / self.aspl_norm/2
            individual.set_fitness(centralities_sum, lengths_sum)

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

    def mutate_all(self):
        for individual in self.population:
            chance = uniform(0, 1)
            if chance >= self.mrate:
                individual.mutate()

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

    def get_best_individual(self):
        return max(self.population, key=lambda x: x.fitness)


def main():
    base_graph = getGraph(getDict("cleaned_graphs/1608751820-graph.json"))
    # base_graph=range(2000)
    all_shortest_path_lengths = save_load_shortest_path_lengths(base_graph)
    betweenness_centralities = save_load_betweenness_centralities(base_graph)
    genetic_algorithm = GeneticAlgorithm(base_graph=base_graph,
                                         num_edges=100,
                                         popsize=10,
                                         num_generations=20000,
                                         mrate=0.02,
                                         aspl_dict=all_shortest_path_lengths,
                                         btwn_dict=betweenness_centralities
                                         )
    genetic_algorithm.run()
    print(genetic_algorithm.best_individual)


if __name__ == '__main__':
    main()

"""
The idea is that we have a single node in the network
we want to add edges between this node and other nodes 
such that we maximize the sum of the betweenness centralities of all our neighbors
and the sum of the lengths of all the shortest paths of our neighbors 

These objectives are heuristics for what we actually want. We can generate
several candidates with this method quickly (hopefully) and then evaluate them
with the actual objectives, which takes more time. 


edge information have a bitstring of length equal to 
the number of nodes we are considering opening channels with
0 = our node does not share a channel with this node
1 = our node does share a channel with this nodes  
"""

"""
find the diameter
make a channel between them
repeat

then add a single channel to the node with highest betweenness
"""
