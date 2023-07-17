import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle, sample, choices, random
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from Individuo import Individuo, Point


class Evolution:
    Populacao = List[Individuo]
    def __init__(
        self,
        origens: Dict[int, Point], transbordos: Dict[int, Point],
        portos: Dict[int, Point], clientes: Dict[int, Point],
        cost_matrix: Dict[Tuple[int, int], float],
    ) -> None:
        self.origens = deepcopy(origens)
        self.transbordos = deepcopy(transbordos)
        self.portos = deepcopy(portos)
        self.clientes = deepcopy(clientes)
        self.cost_matrix = deepcopy(cost_matrix)

    def GenInd(self) -> Individuo:
        new_ind = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
        new_ind.give_random_stuff(cost_matrix=self.cost_matrix)
        return new_ind

    def GenPop(self, tam: int) -> Populacao:
        return [self.GenInd() for _ in range(tam)]

    def SelectNew(self, population: Populacao, worst: int) -> Populacao:
        return choices(
            population=population,
            weights=[1 + worst - ind.get_fitness() for ind in population],
            k = 2
        )

    def CrossOver(self, parent_1: Individuo, parent_2: Individuo) -> Individuo:
        pass

    def Mutation(self, individuo: Individuo) -> Individuo:
        for cromo in individuo:
            qnt = self.origens[cromo.gene_id]
            cromo.gene_cap = qnt
            if cromo.conection_type == 'direct_link':
                # Reset stuff
                for key_1, conect in cromo.adjlist:
                    for key_2, value in conect:
                        individuo.portos[key_2] += value[1]
                cromo.adjlist = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))

                #add to transhipment
                cromo = individuo.give_tranship(cromo=cromo, cost_matrix=self.cost_matrix)
            else:
                # Reset stuff
                for key_1, conect in cromo.adjlist:
                    for key_2, value in conect:
                        if key_2 in individuo.portos:
                            individuo.portos[key_2] += value[1]
                        else:
                            individuo.transbordos[key_2] += value[1]
                cromo.adjlist = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
        return individuo

    def run_evo(
        self,
        tam_population: int = 5,
        generation_limit: int = 10,
        show_progress: bool = False,
        croosover_point: int = 0.35,
        mutation_point: int = 0.35
    ) -> None:
        population = self.GenPop(tam=tam_population)
        for i in range(generation_limit):
            population = sorted(
                population,
                key=lambda x: x.get_fitness()
            )
            print(f"population {i}")
            for ind in population:
                print(ind)
            print()
