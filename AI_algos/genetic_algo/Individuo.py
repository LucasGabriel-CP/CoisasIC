import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle, sample, choice, choices, random
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
        self.adjlist[u][v] += deepcopy(weight)
        # breakpoint()

    def get_cost(self) -> float:
        sum = 0
        for _, i in self.adjlist.items():
            for _, j in i.items():
                sum += j[0]
        return sum
    
    def get_total(self) -> float:
        sum = 0
        for _, i in self.adjlist[self.gene_id].items():
            sum += i[1]
        return sum


    def __str__(self) -> str:
        return "id: " + str(self.gene_id) + '\nconection: ' + self.type_conection + '\nCost: ' + str(self.get_cost()) + '\n' + str(self.gene_point)

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
        pt, tb = 0, 0
        new_dna = []
        key_port, key_transbordos = list(self.portos.keys()), list(self.transbordos.keys())
        for cromo in self.cromossomos:
            new_cromo = cromo
            while new_cromo.gene_point.cap != 0:
                if random() < 0.5 and tb < self.k:
                    # send to port
                    id_trans = choice(key_transbordos)
                    while self.transbordos[id_trans].cap == 0:
                        id_trans = choice(key_transbordos)
                    id_port = choice(key_port)
                    while self.portos[id_port].cap == 0:
                        id_port = choice(key_port)
                    cost_trans = cost_matrix[cromo.gene_id][id_trans]
                    cost_port = cost_matrix[id_trans][id_port]
                    qnt_to_transport = min(cromo.gene_point.cap, self.transbordos[id_trans].cap, self.portos[id_port].cap)
                    new_cromo.add_edge(cromo.gene_id, id_trans, np.array([cost_trans * qnt_to_transport, qnt_to_transport]))
                    new_cromo.add_edge(id_trans, id_port, np.array([cost_port * qnt_to_transport, qnt_to_transport]))
                    new_cromo.gene_point -= qnt_to_transport
                    self.portos[id_port] -= qnt_to_transport
                    self.transbordos[id_trans] -= qnt_to_transport
                    tb += 1
                else:
                    #send to trans them to port
                    id_port = choice(key_port)
                    while self.portos[id_port].cap == 0:
                        id_port = choice(key_port)
                    cost = cost_matrix[cromo.gene_id][id_port]
                    qnt_to_transport = min(cromo.gene_point.cap, self.portos[id_port].cap)
                    new_cromo.add_edge(cromo.gene_id, id_port, np.array([cost * qnt_to_transport, qnt_to_transport]))
                    new_cromo.gene_point -= qnt_to_transport
                    self.portos[id_port] -= qnt_to_transport
            new_dna.append(new_cromo)
        self.cromossomos = new_dna

    def give_not_so_random_stuff(self, cost_matrix) -> None:
        new_dna = []
        for cromo in self.cromossomos:
            new_cromo = cromo
            while new_cromo.gene_point.cap != 0:
                if random() < 0.5:
                    # send to port
                    new_cromo = self.give_direct(cromo=new_cromo, cost_matrix=cost_matrix)
                else:
                    #send to trans them to port
                    new_cromo = self.give_tranship(cromo=new_cromo, cost_matrix=cost_matrix)
            new_dna.append(new_cromo)
        self.cromossomos = new_dna

    def give_direct(self, cromo: Cromossomo, cost_matrix) -> Cromossomo:
        [cost, id_port, qnt_to_transport] = self.get_nearest_porto(point=cromo.gene_id,
                                                                    oferta=cromo.gene_point.cap,
                                                                    cost_matrix=cost_matrix)
        cromo.add_edge(cromo.gene_id, id_port, np.array([cost * qnt_to_transport, qnt_to_transport]))
        cromo.gene_point -= qnt_to_transport
        self.portos[id_port] -= qnt_to_transport
        if cromo.type_conection != "none" and cromo.type_conection != 'direct_link':
            cromo.type_conection = "multi"
        else:
            cromo.type_conection = "direct_link"

        return cromo

    def give_tranship(self, cromo: Cromossomo, cost_matrix) -> Cromossomo:
        [cost_trans, id_trans, cost_port, id_port, qnt_to_transport] = self.get_nearest_transbordo(point=cromo.gene_id,
                                                                                                    oferta=cromo.gene_point.cap,
                                                                                                    cost_matrix=cost_matrix)
        if id_trans == -1:
            return self.give_direct(cromo=cromo, cost_matrix=cost_matrix)

        cromo.add_edge(cromo.gene_id, id_trans, np.array([cost_trans * qnt_to_transport, qnt_to_transport]))
        cromo.add_edge(id_trans, id_port, np.array([cost_port * qnt_to_transport, qnt_to_transport]))
        cromo.gene_point -= qnt_to_transport
        self.portos[id_port] -= qnt_to_transport
        self.transbordos[id_trans] -= qnt_to_transport
        if cromo.type_conection != "none" and cromo.type_conection != 'transhipment':
            cromo.type_conection = "multi"
        else:
            cromo.type_conection = "transhipment"

        return cromo

    def get_nearest_porto(self, point, oferta, cost_matrix):
        closest = float('inf')
        id = None
        for key, value in self.portos.items():
            assert(point not in self.portos)
            if value.cap != 0 and cost_matrix[point][key] < closest:
                closest = cost_matrix[point][key]
                id = key
        assert id in self.portos, f'id: {id}'
        return [closest, id, min(oferta, self.portos[id].cap)]
    
    def get_nearest_transbordo(self, point, oferta, cost_matrix):
        cost_trans, cost_port = float('inf'), float('inf')
        id_trans, id_port = None, None
        qnt_to_transport = oferta
        for k, cap_k in self.transbordos.items():
            for j, cap_j in self.portos.items():
                if cap_k.cap > 0 and cap_j.cap > 0 and cost_matrix[point][k] + cost_matrix[k][j] < cost_trans + cost_port:
                    cost_trans = cost_matrix[point][k]
                    cost_port = cost_matrix[k][j]
                    qnt_to_transport = min(qnt_to_transport, cap_k.cap, cap_j.cap)
                    id_trans = k
                    id_port = j
        
        if id_trans is None:
            return [-1]*5

        assert id_trans in self.transbordos, 'id_trans not found'
        assert id_port in self.portos, 'id_port not found'
                
        return [cost_trans, id_trans, cost_port, id_port, qnt_to_transport]

    def get_fitness(self) -> float:
        sum = 0
        for cromo in self.cromossomos:
            sum += cromo.get_cost()
        return sum
    
    def update_info(self) -> None:
        for cromo in self.cromossomos:
            for key_1, value_1 in cromo.adjlist.items():
                qnt = 0
                for key_2, value_2 in value_1.items():
                    qnt += value_2[1]
                    if key_2 in self.portos:
                        self.portos[key_2] -= value_2[1]
                if key_1 in self.transbordos:
                    self.transbordos[key_1] -= qnt

    def __str__(self) -> str:
        ans = '-'*43
        for cromo in self.cromossomos:
            ans += '\n' + str(cromo) + '\n'
            ans += '-'*43
        ans += f'\nfitness: {self.get_fitness()}\n'
        return ans