from helpers import getGraph, getDict
from random import sample, randrange, uniform
from tqdm import tqdm


class Individual:
    def __init__(self, bitstring, o_indices, z_indices):
        self.bitstring = bitstring
        self.o_indices = o_indices
        self.z_indices = z_indices
        self.fitness = 0

    def get_fitness(self):
        # this will probably not be used so we don't have to make a copy of the base graph for every individual
        self.fitness = sum(self.bitstring[:4])  # should end up with ones at the beginning

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
        return "Best Fitness: {}".format(self.fitness)


class GeneticAlgorithm:
    def __init__(self, base_graph, num_edges, num_parents, popsize, num_generations, mrate):
        self.base_graph = base_graph
        self.num_edges = num_edges
        self.num_parents = num_parents
        self.graph_size = len(self.base_graph)
        self.popsize = popsize
        self.num_generations = num_generations
        self.mrate = mrate
        self.population = []
        self.best_individual = Individual(None, None, None)

    def run(self):
        self.populate()
        pbar = tqdm(range(self.num_generations))
        for i in pbar:
            self.gen_num = i
            self.repopulate()
            pbar.set_postfix({"Best Fitness": self.best_individual.fitness})

    def populate(self):
        for i in range(self.popsize):
            bitstring = [0 for i in range(self.graph_size)]
            z_indices = list(range(self.graph_size))
            o_indices = []
            for i in range(self.num_edges):
                i = randrange(len(z_indices))
                z_indices[i], z_indices[-1] = z_indices[-1], z_indices[i]
                o_indices.append(z_indices.pop())
            for index in o_indices:
                bitstring[index] = 1
            self.population.append(Individual(bitstring, o_indices, z_indices))

    def repopulate(self):
        self.get_fitnesses()
        self.best_individual = self.get_best_individual()
        self.select()
        self.mutate_all()
        self.population.append(self.best_individual)

    def get_fitnesses(self):
        for individual in self.population:
            individual.get_fitness()

    def select(self):
        new_population = []
        self.population = sorted(self.population, key=lambda x: x.fitness)
        parents = self.population[:-self.num_parents]  # lets keep it simple with the elite strategy
        for i in range(self.popsize - 1):
            p1, p2 = sample(parents, 2)
            new_population.append(self.crossover(p1, p2))
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

    def get_best_individual(self):
        return max(self.population, key=lambda x: x.fitness)


def main():
    base_graph = getGraph(getDict("cleaned_graphs/1608751820-graph.json"))
    genetic_algorithm = GeneticAlgorithm(base_graph=base_graph,
                                         num_edges=18,
                                         num_parents=4,
                                         popsize=10,
                                         num_generations=10000,
                                         mrate=0.5)
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
