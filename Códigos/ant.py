# -*- coding: utf-8 -*-

import numpy as np
from random import randrange

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Class ant represents one ant and its path
class Ant(object):
    def __init__(self, start_node, size):
        self.path = [start_node]
        self.of_value = 0.0
        self.distance = 0.0

        # Define nós possíveis de visitar excluindo start_node
        self.possible_nodes = [x for x in range(size) if x != start_node] 
        self.fitness = 0.0
        self.f1score = 0.0


    def calculate_probabilities(self, distances, pheromone, alpha, beta):
        '''
        Function to calculate all probability values
        '''
        probabilities = []
        current_node = self.path[-1]
        for i in self.possible_nodes:
            probabilities.append((pheromone[current_node, i]**alpha)*((1/distances[current_node, i])**beta))

        return probabilities/sum(probabilities)
    

    def chose_next_node(self, distances, pheromone, alpha, beta):
        '''
        Function to chose the next visited node
        '''
        probabilities = self.calculate_probabilities(distances, pheromone, alpha, beta)
        rw = np.random.rand()
        sum_ = 0.0
        for i in range(len(probabilities)):
            sum_ += probabilities[i]
            if sum_ >= rw:
                self.path.append(self.possible_nodes.pop(i))
                break
            

    def build_path(self, distances, pheromone, alpha, beta, num_var, possible_nodes):
        '''
        Function to build a path for an ant
        
        Define de forma aleatória para cada formiga um número de variáveis que fará parte do caminho daquela formiga
        Os valores variam entre os números mínimo e máximo de variáveis a serem exploradas
        '''
        for i in range(num_var-1):
            self.chose_next_node(distances, pheromone, alpha, beta)

        return self.path
            

    def calculate_distance(self, distances):
        '''
        Function to calculate the distance traveled by the ant
        
        '''
        current_node = self.path[0]
        for i in range(0, len(self.path) - 1):
            next_node = self.path[i+1]
            self.distance += distances[current_node, next_node]
            current_node = next_node
            

    def of(self):
        '''
        Function to return the of value
        '''
        return self.fitness


    def compare_path(self, path, colony, ant, nvar, distances, pheromone, alpha, beta, num_var, possible_nodes):
        '''
        Compara se o caminho escolhido para uma formiga possui as mesmas variáveis do caminho de uma formiga já existente.
        Caso seja identificada essa igualdade, é feita uma tentativa de escolher novo caminho
        '''
        list1 = sorted(path)
        for path_ant in colony:
            list2 = sorted(path_ant.path)
            if (list1 == list2) and (len(list2) < nvar[1]-1):
                # Gera novo caminho se o tamanho da lista for menor que o limite superior
                # Se a lista já tiver o númer máximo de variáveis permitido qualquer outra lista do mesmo tamanho
                # terá as mesmas variáveis
                for num_try in range(0, 5):    # Irá tentar até 5x gerar um novo caminho
                # Redefine novo caminho mantendo posição inicial da formiga
                    ant.path = [ant.path[0]]
                    # Cria nova lista de caminhos possíveis removendo a posição de partida da formiga
                    ant_nodes = list(range(0, distances.shape[0]))
                    ant_nodes.remove(ant.path[0])   
                    ant.possible_nodes = ant_nodes
                    # Reconstrói novo caminho
                    path = ant.build_path(distances, pheromone, alpha, beta, num_var, ant.possible_nodes)
                    list1 = sorted(path)
                    if list1 != list2:
                        break
        return path
        



