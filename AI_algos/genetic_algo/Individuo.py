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
        self.adjlist: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
    
    def add_edge(self, u: int, v: int, weight: int) -> None:
        self.adjlist[u][v] += weight

    def get_cost(self) -> float:
        sum = 0
        for _, i in self.adjlist.items():
            for _, j in i.items():
                sum += j
        return sum

class Individuo:
    def __init__(self, _origens: Dict[int, Point], _transbordos: Dict[int, Point], _portos: Dict[int, Point], _clientes: Dict[int, Point]) -> None:
        self.transbordos = deepcopy(_transbordos)
        self.portos = deepcopy(_portos)
        self.clientes = deepcopy(_clientes)

        self.n = len(_origens)
        self.m = len(_portos)
        self.k = len(_transbordos)
        self.o = len(_clientes)

        self.cromossomos = []
        for i in range(self.n):
            self.cromossomos.append(Cromossomo(i, _origens[i]))
        shuffle(self.cromossomos)


    def give_random_stuff(self, cost_matrix):
        for cromo in self.cromossomos:
            while cromo.gene_point.cap != 0:
                if random() < 0.5:
                    # send to port
                    [cost, id_port, qnt_to_transport] = self.get_nearest_porto(point=cromo.gene_id,
                                                                               oferta=cromo.gene_point.cap,
                                                                               cost_matrix=cost_matrix)
                    cromo.gene_point -= qnt_to_transport
                    self.portos[id_port] -= qnt_to_transport
                    cromo.add_edge(cromo.gene_id, id_port, cost)
                else:
                    #send to trans them to port
                    [cost_trans, id_trans, cost_port, id_port, qnt_to_transport] = self.get_nearest_transbordo(point=cromo.gene_id,
                                                                                                                oferta=cromo.gene_point.cap,
                                                                                                                cost_matrix=cost_matrix)
                    cromo.gene_point -= qnt_to_transport
                    self.portos[id_port] -= qnt_to_transport
                    cromo.add_edge(cromo.gene_id, id_port, cost_port)
                    self.transbordos[id_trans] -= qnt_to_transport
                    cromo.add_edge(cromo.gene_id, id_port, cost_trans)


    def get_nearest_porto(self, point, oferta, cost_matrix):
        closest = float('inf')
        id = None
        for key, value in self.portos:
            if value.cap != 0 and cost_matrix[point][key] < closest:
                closest = cost_matrix[point][key]
                id = key
        
        return [closest, id, min(oferta, self.portos[id].cap)]
    
    def get_nearest_transbordo(self, point, oferta, cost_matrix):
        closest_trans = float('inf')
        id_trans = None
        for key, value in self.transbordos:
            if value.cap != 0 and cost_matrix[point][key] < closest_trans:
                closest_trans = cost_matrix[point][key]
                id_trans = key
        
        [cost_port, id_port, qnt_to_transport] = self.get_nearest_porto(point=id_trans,
                                                                        oferta=min(oferta, self.transbordos[id_trans].cap),
                                                                        cost_matrix=cost_matrix)
        
        return [closest_trans, id_trans, cost_port, id_port, qnt_to_transport]
    