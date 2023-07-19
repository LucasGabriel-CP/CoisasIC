import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle, sample, choices, random
from copy import deepcopy
from math import sqrt, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from utils import Individuo, Point

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
        parents = choices(
            population=population,
            weights=[1 + worst - ind.get_fitness() for ind in population],
            k = 7
        )
        parents = sorted(
            parents,
            key=lambda x: x.get_fitness()
        )
        return [parents[0], parents[1]]

    def CrossOver(self, parent_1: Individuo, parent_2: Individuo) -> Individuo:
        child_routes = parent_1.cromossomos[:(len(parent_1.cromossomos) + 1)//2]

        has = {}
        for i, _ in self.origens.items():
            has[i] = False
        for cromo in child_routes:
            has[cromo.gene_id] = True
        
        Children = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
        Children.cromossomos = deepcopy(child_routes)

        
        for cromo in parent_2.cromossomos[::-1]:
            if not has[cromo.gene_id]:
                Children.cromossomos.append(cromo)

        if not Children.update_info():
            aux = Children.cromossomos
            Children = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
            for i in range(len(Children.cromossomos)):
                Children.cromossomos[i].gene_id = aux[i].gene_id
                Children.cromossomos[i].gene_point = self.origens[aux[i].gene_id]
            Children.give_not_so_random_stuff(cost_matrix=self.cost_matrix)

        return Children

    def Mutation(self, individuo: Individuo, best: float) -> Individuo:
        individuo = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
        individuo.give_random_stuff(cost_matrix=self.cost_matrix)
        return individuo

    def run_evo(
        self,
        fitness_limit: float = -1,
        elitism: int = 2,
        tam_population: int = 128,
        generation_limit: int = 1024,
        show_progress: bool = False,
        croosover_point: int = 0.35,
        mutation_point: int = 0.45
    ) -> Tuple[Individuo, float]:
        population = self.GenPop(tam=tam_population)
        st = time.time()
        nd = None
        best = population[0].get_fitness()
        elitism_size = ceil(tam_population * elitism)
        for i in range(generation_limit):
            population = sorted(
                population,
                key=lambda x: x.get_fitness()
            )
            cnt = 0
            for ind in population:
                if ind.get_fitness() != population[0].get_fitness():
                    break
                cnt += 1
            if show_progress:
                print(f"Generation {i}, Fitness {cnt}: {population[0].get_fitness()}")
            if best - population[0].get_fitness() < 1e-6:
                best = population[0].get_fitness()
                nd = time.time()
            if abs(population[0].get_fitness() - fitness_limit) < 1e-6:
                break
            next_gen = deepcopy(population[:elitism_size])
            worst = population[-1].get_fitness()
            for _ in range(elitism_size, tam_population):
                parents = self.SelectNew(population=population, worst=worst)
                if random() < croosover_point:
                    child = self.CrossOver(parent_1=parents[0], parent_2=parents[1])
                    if random() < mutation_point:
                        child = self.Mutation(child, population[0].get_fitness())
                    next_gen.append(deepcopy(child))
                else:
                    if random() < mutation_point:
                        parents[0] = self.Mutation(parents[0], population[0].get_fitness())
                    next_gen.append(deepcopy(parents[0]))
            population = deepcopy(next_gen)
        if nd is None:
            nd = time.time()
        return [population[0], nd - st]
            
