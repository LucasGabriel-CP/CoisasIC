import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle, sample, choice, choices, random
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from .Point import Point

class Cromossomo:
    def __init__(self, _id: int, _point: Point) -> None:
        self.gene_id = _id
        self.gene_point = _point
        self.type_conection = "none"
        self.total_cost = 0
        self.total_send = 0
        self.adjlist = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    
    def add_edge(self, u: int, v: int, weight: np.array) -> None:
        self.total_cost += weight[0]
        if u == self.gene_id:
            self.total_send += weight[1]
        assert self.total_send >= 0, f'total send: {self.total_send}, weight: {weight}'
        self.adjlist[u][v] += deepcopy(weight)
        # breakpoint()

    def __str__(self) -> str:
        return "id: " + str(self.gene_id) + '\nconection: ' + self.type_conection + '\nSend: ' + str(self.total_send) + '\nCost: ' + str(self.total_cost) + '\n' + str(self.gene_point)

class Individuo:
    def __init__(self, origens: Dict[int, Point], transbordos: Dict[int, Point], portos: Dict[int, Point], demand: float) -> None:
        self.transbordos = deepcopy(transbordos)
        self.portos = deepcopy(portos)
        self.rank = None
        self.n =list(origens.keys())
        self.m = list(portos.keys())
        self.k = list(transbordos.keys())
        
        self.demand = demand

        self.cromossomos = []
        for i in range(len(self.n)):
            self.cromossomos.append(Cromossomo(i, origens[i]))
        shuffle(self.cromossomos)

    # def reverse_cromo(self, id1: int, id2: int) -> None:
    #     self.cromossomos = self.cromossomos[:id1] + self.cromossomos[id2:id1 - 1:-1] + self.cromossomos[id2 + 1:]
    #     new_ind.give_not_so_random_stuff(cost_matrix=self.cost_matrix)

    def give_random_stuff(self, cost_matrix) -> None:
        if random() < 0.5:
            self.give_not_so_random_stuff(cost_matrix=cost_matrix)
            return
        new_dna = []
        key_port, key_transbordos = list(self.portos.keys()), list(self.transbordos.keys())
        tot = 0
        for cromo in self.cromossomos:
            new_cromo = cromo
            while new_cromo.gene_point.cap != 0 and self.demand - tot - new_cromo.total_send > 1e-9:
                op = randint(self.k[0], self.m[-1])
                
                if op in self.transbordos:
                    id_trans = op
                    while self.transbordos[id_trans].cap == 0:
                        id_trans = choice(key_transbordos)
                    id_port = choice(key_port)
                    while self.portos[id_port].cap == 0:
                        id_port = choice(key_port)
                    cost_trans = cost_matrix[cromo.gene_id][id_trans]
                    cost_port = cost_matrix[id_trans][id_port]
                    qnt_to_transport = min(cromo.gene_point.cap, self.transbordos[id_trans].cap, self.portos[id_port].cap, self.demand - tot - new_cromo.total_send)
                    assert qnt_to_transport <= self.portos[id_port].cap and qnt_to_transport <= self.transbordos[id_trans].cap

                    new_cromo.add_edge(cromo.gene_id, id_trans, np.array([cost_trans * qnt_to_transport, qnt_to_transport]))
                    new_cromo.add_edge(id_trans, id_port, np.array([cost_port * qnt_to_transport, qnt_to_transport]))
                    new_cromo.gene_point -= qnt_to_transport
                    
                    self.portos[id_port] -= qnt_to_transport
                    self.transbordos[id_trans] -= qnt_to_transport
                else:
                    id_port = op
                    while self.portos[id_port].cap == 0:
                        id_port = choice(key_port)
                    cost = cost_matrix[cromo.gene_id][id_port]
                    qnt_to_transport = min(cromo.gene_point.cap, self.portos[id_port].cap, self.demand - tot - new_cromo.total_send)
                    assert qnt_to_transport <= self.portos[id_port].cap

                    new_cromo.add_edge(cromo.gene_id, id_port, np.array([cost * qnt_to_transport, qnt_to_transport]))
                    new_cromo.gene_point -= qnt_to_transport
                    
                    self.portos[id_port] -= qnt_to_transport
            tot += new_cromo.total_send
            new_dna.append(new_cromo)
        self.cromossomos = new_dna

    def give_not_so_random_stuff(self, cost_matrix) -> None:
        new_dna = []
        tot = 0
        for cromo in self.cromossomos:
            new_cromo = cromo
            while new_cromo.gene_point.cap != 0 and self.demand - tot - new_cromo.total_send > 1e-9:
                op = randint(self.k[0], self.m[-1])
                if op in self.transbordos:
                    new_cromo = self.give_tranship(cromo=new_cromo, cost_matrix=cost_matrix, tot=tot+new_cromo.total_send)
                else:
                    new_cromo = self.give_direct(cromo=new_cromo, cost_matrix=cost_matrix, tot=tot+new_cromo.total_send)
            new_dna.append(new_cromo)
            tot += new_cromo.total_send
            assert(new_cromo.gene_id == cromo.gene_id)
        self.cromossomos = new_dna

    def give_direct(self, cromo: Cromossomo, cost_matrix, tot: float) -> Cromossomo:
        [cost, id_port, qnt_to_transport] = self.get_nearest_porto(point=cromo.gene_id,
                                                                    oferta=cromo.gene_point.cap,
                                                                    cost_matrix=cost_matrix,
                                                                    tot=tot)
        cromo.add_edge(cromo.gene_id, id_port, np.array([cost * qnt_to_transport, qnt_to_transport]))
        cromo.gene_point -= qnt_to_transport
        self.portos[id_port] -= qnt_to_transport

        if cromo.type_conection != "none" and cromo.type_conection != 'direct_link':
            cromo.type_conection = "multi"
        else:
            cromo.type_conection = "direct_link"

        return cromo

    def give_tranship(self, cromo: Cromossomo, cost_matrix, tot: float) -> Cromossomo:
        [cost_trans, id_trans, cost_port, id_port, qnt_to_transport] = self.get_nearest_transbordo(point=cromo.gene_id,
                                                                                                    oferta=cromo.gene_point.cap,
                                                                                                    cost_matrix=cost_matrix,
                                                                                                    tot=tot)
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

    def get_nearest_porto(self, point, oferta, cost_matrix, tot) -> Tuple[float, int, float]:
        closest = float('inf')
        id = None
        for key, value in self.portos.items():
            assert(point not in self.portos)
            if value.cap != 0 and cost_matrix[point][key] < closest:
                closest = cost_matrix[point][key]
                id = key
        assert id in self.portos, f'id: {id}'
        return [closest, id, min(oferta, self.portos[id].cap, self.demand - tot)]
    
    def get_nearest_transbordo(self, point, oferta, cost_matrix, tot) -> Tuple[float, int, float, int, int]:
        cost_trans, cost_port = float('inf'), float('inf')
        id_trans, id_port = None, None
        qnt_to_transport = oferta
        for k, cap_k in self.transbordos.items():
            for j, cap_j in self.portos.items():
                if cap_k.cap > 0 and cap_j.cap > 0 and cost_matrix[point][k] + cost_matrix[k][j] < cost_trans + cost_port:
                    cost_trans = cost_matrix[point][k]
                    cost_port = cost_matrix[k][j]
                    # breakpoint()
                    qnt_to_transport = min(qnt_to_transport, cap_k.cap, cap_j.cap, self.demand - tot)
                    id_trans = k
                    id_port = j
        
        if id_trans is None:
            return [-1]*5

        assert id_trans in self.transbordos, 'id_trans not found'
        assert id_port in self.portos, 'id_port not found'
        assert qnt_to_transport <= self.portos[id_port].cap and qnt_to_transport <= self.transbordos[id_trans].cap

        return [cost_trans, id_trans, cost_port, id_port, qnt_to_transport]

    def get_fitness(self) -> float:
        sum = 0
        for cromo in self.cromossomos:
            sum += cromo.total_cost
        return sum
    
    def get_total(self) -> float:
        sum = 0
        for cromo in self.cromossomos:
            sum += cromo.total_send
        return sum
    
    def check_info(self) -> bool:
        for cromo in self.cromossomos:
            if cromo.gene_point.cap < .0:
                return False
            for key_1, value_1 in cromo.adjlist.items():
                for key_2, value_2 in value_1.items():
                    if key_2 in self.portos:
                        if self.portos[key_2].cap < .0:
                            return False
                    elif key_2 in self.transbordos:
                        if self.transbordos[key_2].cap < .0:
                            return False
        return abs(self.demand - self.get_total()) < 1e-6
    
    def __str__(self) -> str:
        ans = '-'*43
        for cromo in self.cromossomos:
            ans += '\n' + str(cromo) + '\n'
            ans += '-'*43
        ans += f'\nfitness: {self.get_fitness()}\n'
        return ans