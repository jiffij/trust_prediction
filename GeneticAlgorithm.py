import numpy as np
from dataclasses import dataclass
from random import randint, random, uniform, sample
import math
from copy import deepcopy as dc


@dataclass
class Chromosome:
    """
    :param num_layer: the number of layer
    :param max_node: a list of maximum possible number of nodes in each layer
    :param node_percent: a list of percentage (0,1] multiply with max_node
    and floor to get the actual number of node in a layer
    """
    num_layer: int
    max_node: list
    node_percent: list
    RMSE: float
    fitness_score: float
    tan: int
    id: int

    def __str__(self):
        # return f"Chromosome(lambda={self.max_node}, delta={self.node_percent})"
        layers = [math.ceil(max_node * percent) for max_node, percent in zip(self.max_node, self.node_percent)]
        return f"Chromosome(id={self.id}, layers={layers}, score={self.fitness_score})"

    def update_RMSE(self, RMSE):
        self.RMSE = RMSE


class GeneticAlgorithm:
    def __init__(self, population, winning_percentage=0.1, layer_range=(2, 6), node_range=(32, 256),
                 mutation_rate=0.3):
        """
        This class handles all activities of the Genetic Algorithm, including mutation, crossover, initialization...
        Make sure the initial population is large enough to not become too homogenous
        :param population:
        :param winning_percentage:
        :param layer_range:
        :param node_range:
        """
        self.population = population
        self.winning_percentage = winning_percentage
        self.layer_range = layer_range
        self.node_range = node_range
        self.mu = mutation_rate
        self.epoch = 0
        self.history = ""
        self.chromosomes = list()
        self.chromosome_generator()
        self.history += str(self)
        print(str(self))

    def __str__(self):
        message = f"Genetic Algorithm Epoch {self.epoch}: \n"
        for chromosome in self.chromosomes:
            message += str(chromosome) + ", "
        message += "\n"
        return message

    def save_history(self):
        """
        This function save the GA history to a txt file
        :return:
        """
        with open("GA.txt", "w") as text_file:
            text_file.write(self.history)

    def chromosome_generator(self):
        """
        randomly generate chromosomes
        :return:
        """
        for i in range(self.population):
            num_layer = randint(self.layer_range[0], self.layer_range[1])
            max_node = [randint(self.node_range[0], self.node_range[1]) for i in range(num_layer)]
            node_percent = [uniform(0.0001, 1) for i in range(num_layer)]
            rmse = -1
            # DEBUG
            # rmse = randint(1, 100)
            self.chromosomes.append(Chromosome(num_layer=num_layer, max_node=max_node, node_percent=node_percent,
                                               RMSE=rmse, fitness_score=0, tan=0, id=i))

    @staticmethod
    def crossover(chromosome_a, chromosome_b):
        beta = random()
        p = chromosome_a.num_layer / chromosome_b.num_layer
        if p == 1:  # same number of layer
            p = 0
        elif p < 1:  # chromosome_b has more layer that chromosome_a, reassign b to a
            chromosome_a, chromosome_b = chromosome_b, chromosome_a
            p = chromosome_a.num_layer / chromosome_b.num_layer
        for i in range(chromosome_b.num_layer):  # either they have same # of layer or a > b
            selected_layer = randint(i, math.ceil(i + p))  # select a layer in A to crossover with b
            selected_layer = min(chromosome_a.num_layer-1, selected_layer)
            # Crossover between layer
            delta_a, delta_b = chromosome_a.node_percent[i], chromosome_b.node_percent[i]
            lamda_a, lamda_b = bin(chromosome_a.max_node[i])[2:], bin(chromosome_b.max_node[i])[2:]
            # integer crossover
            chromosome_a.node_percent[selected_layer] = round(beta * delta_a + (1 - beta) * delta_b)
            chromosome_b.node_percent[i] = round(beta * delta_b + (1 - beta) * delta_a)
            # single point crossover
            k = randint(1, min(len(lamda_a), len(lamda_b)) - 1)
            chromosome_a.max_node[selected_layer] = int(lamda_a[:k] + lamda_b[k:], 2)
            chromosome_b.max_node[i] = int(lamda_b[:k] + lamda_a[k:], 2)
        return chromosome_a, chromosome_b

    def mutation(self, chromosome):
        """
        This function mutate the max_node and node_percent with ratio mu_lamda and mu_delta
        :param chromosome: the target Chromosome
        :return: the mutated Chromosome itself
        """
        mu_delta = uniform(0, self.mu)
        mu_lamda = uniform(0, self.mu)
        for i in range(chromosome.num_layer):
            # mutate the lambda
            lamda = bin(chromosome.max_node[i])[2:]
            max_bit = min(len(lamda) - 1, math.ceil(math.log2(self.node_range[1])))
            indices = sample(range(0, max_bit), math.ceil(max_bit * mu_lamda))
            lamda = list(lamda)
            for x in indices:
                lamda[x] = '0' if lamda[x] == '1' else '1'
            # mutate delta
            mutated_delta = chromosome.node_percent[i] + uniform(-mu_delta, mu_delta)
            mutated_delta = max(0, min(1, mutated_delta))
            chromosome.node_percent[i] = mutated_delta
        return chromosome

    def update_fitness(self):
        tan_0 = 1000
        for chromosome in self.chromosomes:
            tan = 0
            for i in range(chromosome.num_layer-1):
                if math.ceil(chromosome.max_node[i] * chromosome.node_percent[i]) <= math.ceil(
                        chromosome.max_node[i + 1] * chromosome.node_percent[i + 1]):
                    tan += 1
            chromosome.tan = tan
            tan_0 = min(tan, tan_0)
        for chromosome in self.chromosomes:
            phi = chromosome.tan / (chromosome.num_layer - 1) + tan_0
            chromosome.fitness_score = phi / chromosome.RMSE

    def selection(self, breeding_ratio=0.6):
        """
        This function separate the winning genes and the rest, also select chromosomes with higher fitness score for
        breeding, breeding set better be even number
        :param breeding_ratio: this is the ratio for selecting parents for crossover in the non-winning population
        :return: list of winning chromosomes, breeding, non-breeding chromosomes
        """
        chromosomes = dc(self.chromosomes)
        sorted_chromosomes = sorted(chromosomes, key=lambda item: item.fitness_score, reverse=True)
        winning_chromosomes = sorted_chromosomes[:math.ceil(self.population * 0.1)]
        breeding_chromosomes = sorted_chromosomes[math.ceil(self.population * 0.1):]
        weight = np.array([i.fitness_score for i in breeding_chromosomes])
        weight = weight / np.sum(weight)
        parents = np.random.choice(np.arange(len(breeding_chromosomes)),
                                   size=math.ceil(len(breeding_chromosomes) * breeding_ratio) +
                                        (1 if math.ceil(len(breeding_chromosomes) * breeding_ratio) % 2 == 1 else 0),
                                   replace=False, p=weight)
        single = np.array([i for i in range(len(breeding_chromosomes)) if i not in parents])
        non_breeding_chromosomes = [breeding_chromosomes[i] for i in single]
        breeding_chromosomes = [breeding_chromosomes[i] for i in parents]
        return winning_chromosomes, breeding_chromosomes, non_breeding_chromosomes

    @staticmethod
    def pair_up(breeding_size):
        """
        This function return random index pair
        :param breeding_size:
        :return: a list of index pair
        """
        index = np.arange(breeding_size)
        index = np.random.permutation(index)
        return [(index[i], index[i + 1]) for i in range(0, breeding_size, 2)]

    def step(self):
        self.update_fitness()
        winning_chromosomes, breeding_chromosomes, non_breeding_chromosomes = self.selection()
        pair_indices = self.pair_up(len(breeding_chromosomes))
        for pair in pair_indices:
            breeding_chromosomes[pair[0]], breeding_chromosomes[pair[1]] = \
                self.crossover(breeding_chromosomes[pair[0]], breeding_chromosomes[pair[1]])
        non_winning_chromosomes = breeding_chromosomes + non_breeding_chromosomes
        for i in range(len(non_winning_chromosomes)):
            non_winning_chromosomes[i] = self.mutation(non_winning_chromosomes[i])
        new_chromosomes = winning_chromosomes + non_winning_chromosomes
        self.epoch += 1
        self.chromosomes = new_chromosomes
        self.history += str(self)
        print(str(self))
