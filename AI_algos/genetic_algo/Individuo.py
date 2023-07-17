import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle, sample, choices, random
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from utils.Point import Point

class Cromossomo:
    def __init__(self, _id: int, _point: Point) -> None:
        self.gene_id = _id
        self.gene_point = _point
        self.type_conection = "none"
        self.adjlist = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    
    def add_edge(self, u: int, v: int, weight: np.array) -> None:
        self.adjlist[u][v] += weight

    def get_cost(self) -> float:
        sum = 0
        for _, i in self.adjlist.items():
            for _, j in i.items():
                sum += j[0]
        return sum
    
    def __str__(self) -> str:
        return "id: " + str(self.gene_id) + '\n' + str(self.gene_point)

class Individuo:
    def __init__(self, origens: Dict[int, Point], transbordos: Dict[int, Point], portos: Dict[int, Point], clientes: Dict[int, Point]) -> None:
        self.transbordos = deepcopy(transbordos)
        self.portos = deepcopy(portos)
        self.clientes = deepcopy(clientes)

        self.n = len(origens)
        self.m = len(portos)
        self.k = len(transbordos)
        self.o = len(clientes)

        self.cromossomos = []
        for i in range(self.n):
            self.cromossomos.append(Cromossomo(i, origens[i]))
        shuffle(self.cromossomos)

    def give_random_stuff(self, cost_matrix) -> None:
        for cromo in self.cromossomos:
            while cromo.gene_point.cap != 0:
                if random() < 0.5:
                    # send to port
                    cromo = self.give_direct(cromo=cromo, cost_matrix=cost_matrix)
                else:
                    #send to trans them to port
                    cromo = self.give_tranship(cromo=cromo, cost_matrix=cost_matrix)

    def give_direct(self, cromo: Cromossomo, cost_matrix) -> Cromossomo:
        [cost, id_port, qnt_to_transport] = self.get_nearest_porto(point=cromo.gene_id,
                                                                    oferta=cromo.gene_point.cap,
                                                                    cost_matrix=cost_matrix)
        cromo.gene_point -= qnt_to_transport
        self.portos[id_port] -= qnt_to_transport
        cromo.add_edge(cromo.gene_id, id_port, np.array([cost, qnt_to_transport]))
        cromo.type_conection = "direct_link"

        return cromo

    def give_tranship(self, cromo: Cromossomo, cost_matrix) -> Cromossomo:
        [cost_trans, id_trans, cost_port, id_port, qnt_to_transport] = self.get_nearest_transbordo(point=cromo.gene_id,
                                                                                                    oferta=cromo.gene_point.cap,
                                                                                                    cost_matrix=cost_matrix)
        cromo.gene_point -= qnt_to_transport
        self.portos[id_port] -= qnt_to_transport
        cromo.add_edge(cromo.gene_id, id_port, np.array([cost_port, qnt_to_transport]))
        self.transbordos[id_trans] -= qnt_to_transport
        cromo.add_edge(cromo.gene_id, cromo.gene_id, np.array([cost_trans, qnt_to_transport]))
        cromo.type_conection = "transhipment"

        return cromo

    def get_nearest_porto(self, point, oferta, cost_matrix):
        closest = float('inf')
        id = None
        for key, value in self.portos.items():
            if value.cap != 0 and cost_matrix[point, key] < closest:
                closest = cost_matrix[point, key]
                id = key
        
        return [closest, id, min(oferta, self.portos[id].cap)]
    
    def get_nearest_transbordo(self, point, oferta, cost_matrix):
        closest_trans = float('inf')
        id_trans = None
        for key, value in self.transbordos.items():
            if value.cap != 0 and cost_matrix[point, key] < closest_trans:
                closest_trans = cost_matrix[point, key]
                id_trans = key
        
        [cost_port, id_port, qnt_to_transport] = self.get_nearest_porto(point=id_trans,
                                                                        oferta=min(oferta, self.transbordos[id_trans].cap),
                                                                        cost_matrix=cost_matrix)
        
        return [closest_trans, id_trans, cost_port, id_port, qnt_to_transport]

    def get_fitness(self) -> float:
        sum = 0
        for cromo in self.cromossomos:
            sum += cromo.get_cost()
        return sum
    
    def __str__(self) -> str:
        ans = '-'*43
        for cromo in self.cromossomos:
            ans += '\n' + str(cromo) + '\n'
            ans += '-'*43
        ans += f'\nfitness: {self.get_fitness()}\n'
        return ans