import os

import numpy as np
from dataclasses import dataclass
from random import randint, random, uniform, sample
import math
from copy import deepcopy as dc
import datetime
import pickle
from pathlib import Path


@dataclass
class Chromosome:
    """
    The chromosome class implement the unit of one individual in the Genetic Algorithm. It contains information
    including: number of layer, number of maximum possible node in each layer (lambda), the sigma that represent the
    percentage of actual number of nodes allowed ceil(lambda x sigma), the root mean square error of the chromosome,
    the fitness score, the knowledge abstraction level, and the id of the chromosome

    :param num_layer: the number of layer
    :param max_node: (lambda) a list of maximum possible number of nodes in each layer
    :param node_percent: (sigma) a list of percentage (0,1] multiply with (lambda) to get the actual number of node and floor to get the actual number of node in a layer
    :param RMSE: the root mean square error
    :param fitness_score: The fitness score
    :param tan: the knowledge abstraction level
    :param id: the id of the chromosome
    """
    num_layer: int
    max_node: list
    node_percent: list
    RMSE: float
    fitness_score: float
    tan: int
    id: int

    def __str__(self):
        """
        This print useful information of a chromosome, i.e. Chromosome(id=7, layers=[174, 82], RMSE=0.24830233520916337, score=4.027348349976759)

        :return: useful information as string
        """
        layers = self.get_layers()
        return f"Chromosome(id={self.id}, layers={layers}, RMSE={self.RMSE}, score={self.fitness_score})"

    def update_RMSE(self, RMSE):
        """
        This function update the RMSE of the chromosome.

        :param RMSE: new RMSE
        """
        self.RMSE = RMSE


    def zero_exist(self):
        """
        This funciton check if there is any layer contains zero node.

        :return: True if there is zero node
        """
        layers = self.get_layers()
        if 0 in layers:
            return True
        return False

    def get_layers(self):
        """
        This function calculates the number of nodes in each layers by: ceil(lambda x sigma).

        :return: the list containing number of nodes for each layer
        """
        return [int(math.ceil(max_node * percent)) for max_node, percent in zip(self.max_node, self.node_percent)]


class GeneticAlgorithm:
    def __init__(self, population=50, winning_percentage=0.1, layer_range=(2, 6), node_range=(32, 256),
                 mutation_rate=0.3, store_path="./GA", save_freq=3):
        """
        This class handles all activities of the Genetic Algorithm, including mutation, crossover, initialization,
        calculation of fitness score, selection for crossover and mutation, selecting winner group.

        To make sure the initial population is large enough to not become too homogenous, population is set to 100 by default.

        :param population: The initial population of the GA algorithm
        :param winning_percentage: The percentage of winner, which remains unchanged, in each generation
        :param layer_range: The span of possible number of layers for each chromosome
        :param node_range: The span of possible number of nodes for each MLP linear layer
        :param mutation_rate: The probability of each chromosome being mutated
        :param store_path: The path to store the GA checkpoint
        :param save_freq: The frequency to save a checkpoint
        """
        self.population = population
        self.winning_percentage = winning_percentage
        self.layer_range = layer_range
        self.node_range = node_range
        self.mu = mutation_rate
        self.store_path = store_path
        self.epoch = 0
        # self.history = ""
        self.start_time = datetime.datetime.now()
        self.start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "-")
        self.save_freq = save_freq
        self.chromosomes = list()
        self.chromosome_generator()
        # self.history += str(self)
        # print(str(self))
        now = self.start_time
        with open("GA.txt", "a") as text_file:
            text_file.write(f'=========================={now}==========================\n')

    def __str__(self):
        """
        This function print all chromosome of the latest epoch.

        :return: All chromosome in string
        """
        message = f"Genetic Algorithm Epoch {self.epoch-1}: \n"
        for chromosome in self.chromosomes:
            message += str(chromosome) + ", "
        message += "\n"
        return message

    @staticmethod
    def load(file_path):
        """
        This function load the checkpoint stored as pickle file.
        :param file_path: the relative path to the pickle file, i.e. \"./GA/latest.pkl\"
        :return: a GA object
        """
        if not os.path.exists(file_path):
            print(f"This file/directory {file_path} doesn't exist.")
            return
        with open(file_path, 'rb') as f:
            loaded_object = pickle.load(f)
        return loaded_object

    def save(self, latest=False):
        """
        This function save the checkpoint as a pickle file.
        :param latest: is it the latest file
        """
        now = self.start_time
        now += f'_Epoch-{self.epoch-1}'
        if latest:
            now = "latest"
        Path(self.store_path).mkdir(parents=True, exist_ok=True)
        with open(f'{self.store_path}/{now}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def save_history(self):
        """
        This function save the GA history to a txt file called GA.txt
        """
        # now = datetime.datetime.now()
        # now = now.strftime("%Y-%m-%d %H:%M:%S")
        with open("GA.txt", "a") as text_file:
            text_file.write(str(self))


    def print_history(self):
        print(self.history)

    def chromosome_generator(self):
        """
        The chromosome_generator randomly generate chromosomes with different number of layers and nodes up to
        population
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
        """
        This function performs the crossover operation for the lambda (max_node) and sigma (node_percent) between
        chromosome_a and chromosome_b.

        Since sigma (node_percent) is a list of integer and lambda (max_node) is treated as a list of binary,
        integer crossover and single point crossover is performed on lambda and sigma, respectively.

        Integer crossover

        Single point crossover

        :param chromosome_a: selected chromosome
        :param chromosome_b: selected chromosome
        :return:
        """
        beta = random()
        p = chromosome_a.num_layer / chromosome_b.num_layer
        if p == 1:  # same number of layer
            p = 0
        elif p < 1:  # chromosome_b has more layer that chromosome_a, reassign b to a
            chromosome_a, chromosome_b = chromosome_b, chromosome_a
            p = chromosome_a.num_layer / chromosome_b.num_layer
        for i in range(chromosome_b.num_layer):  # either they have same # of layer or a > b
            selected_layer = randint(i, math.ceil(i + p))  # select a layer in A with range [i,i+p] to crossover with b
            selected_layer = min(chromosome_a.num_layer-1, selected_layer)
            # Crossover between layer
            # get the lambda and sigma
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
        This function mutate the max_node and node_percent with mutation ratio mu_lamda and mu_delta, which have range
        [0, 0.3].
        The lambda is mutated by randomly toggling random number of bit = mu_lamda x max_bit.
        The sigma is mutated by randomly add a random float of range = [-mu_delta, mu_delta]

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
        """
        This function update the fitness score of each chromosome with the new RMSE.
        The fitness score is updated with formula:

        :return:
        """
        tan_0 = 1000
        for chromosome in self.chromosomes:
            tan = 0
            for i in range(chromosome.num_layer-1):
                if math.ceil(chromosome.max_node[i] * chromosome.node_percent[i]) >= math.ceil( # <=
                        chromosome.max_node[i + 1] * chromosome.node_percent[i + 1]):
                    tan += 1
            chromosome.tan = tan
            tan_0 = min(tan, tan_0)
        for chromosome in self.chromosomes:
            phi = chromosome.tan / (chromosome.num_layer - 1) + tan_0
            chromosome.fitness_score = phi / chromosome.RMSE # minimize knowledge abstraction?

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
        weight += 0.1 ## preventing zero case
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
        This function return random index pairs
        :param breeding_size: the size of the breeding group
        :return: a list of index pair
        """
        index = np.arange(breeding_size)
        index = np.random.permutation(index)
        return [(index[i], index[i + 1]) for i in range(0, breeding_size, 2)]

    def winner(self):
        """
        This function sort the chromosomes by fitness score and return the best performing chromosome.

        :return: the chromosome with highest fitness score
        """
        sorted_chromosomes = sorted(self.chromosomes, key=lambda item: item.fitness_score, reverse=True)
        return sorted_chromosomes[0]

    def step(self):
        """
        This function perform all operation for one generation. It first updates the fitness score of each
        chromosome. Then, it separates the population into the winner group (remain unchanged),
        breeding group (crossover and mutation), and non-breeding group (only mutation). Operations are the
        performed, crossover followed mutation. If any chromosome contains zero nodes, it is being mutated until no
        zero layer left.
        """
        # update fitness score
        self.update_fitness()
        # separate the population into the winner group (remain unchanged), breeding group (crossover and mutation),
        # and non-breeding group (only mutation)
        winning_chromosomes, breeding_chromosomes, non_breeding_chromosomes = self.selection()
        # crossover
        pair_indices = self.pair_up(len(breeding_chromosomes))
        for pair in pair_indices:
            breeding_chromosomes[pair[0]], breeding_chromosomes[pair[1]] = \
                self.crossover(breeding_chromosomes[pair[0]], breeding_chromosomes[pair[1]])
        # mutation
        non_winning_chromosomes = breeding_chromosomes + non_breeding_chromosomes
        for i in range(len(non_winning_chromosomes)):
            non_winning_chromosomes[i] = self.mutation(non_winning_chromosomes[i])
        new_chromosomes = winning_chromosomes + non_winning_chromosomes
        # check if there are layer with zero node, mutate it until no zero layer
        for chromosome in new_chromosomes:
            while chromosome.zero_exist():
                self.mutation(chromosome)
        # update and save checkpoints
        self.epoch += 1
        self.chromosomes = new_chromosomes
        if self.epoch % self.save_freq == 0:
            self.save()
        self.save(latest=True)
        self.save_history()
        print(str(self))
