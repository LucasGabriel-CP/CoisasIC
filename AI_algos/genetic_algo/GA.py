import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint, shuffle, choices
from copy import deepcopy
from math import sqrt, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from utils import Individuo, Point, Cromossomo

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
            k = 5
        )
        parents = sorted(
            parents,
            key=lambda x: x.get_fitness()
        )
        return [parents[0], parents[1]]

    def CrossOver(self, parent_1: Individuo, parent_2: Individuo) -> Individuo:
        child_routes = []

        has = {}
        for i, _ in self.origens.items():
            has[i] = False
        for cromo in parent_1.cromossomos[:(len(parent_1.cromossomos) + 1)//2]:
            child_routes.append(cromo)
            has[cromo.gene_id] = False
        
        Children = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
        Children.cromossomos = deepcopy(child_routes)
        
        for cromo in parent_2.cromossomos[::-1]:
            if not has[cromo.gene_id]:
                Children.cromossomos.append(Cromossomo(cromo.gene_id, self.origens[cromo.gene_id]))
        
        Children.give_not_so_random_stuff(cost_matrix=self.cost_matrix)

        if not Children.check_info():
            aux = Children.cromossomos
            Children = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
            for i in range(len(Children.cromossomos)):
                Children.cromossomos[i].gene_id = aux[i].gene_id
                Children.cromossomos[i].gene_point = self.origens[aux[i].gene_id]
            Children.give_not_so_random_stuff(cost_matrix=self.cost_matrix)

        return Children

    def crossover_prob(slef, fmax: float, favg: float, fat: float):
        k3 = 1.0
        k1 = random.uniform(k3, 1.0 + 1e-7)
        if fat < favg:
            return k3
        return k1 * (fmax - favg) / max(fmax - fat, 1)


    def mutation_prob(self, fmax: float, favg: float, fat: float):
        k4 = 0.5
        k2 = random.uniform(k4, 1.0 + 1e-7)
        if fat < favg:
            return k4
        return k2 * (fmax - favg) / max(fmax - fat, 1)

    def Mutation(self, individuo: Individuo) -> Individuo:
        id1, id2 = randint(1, len(individuo.cromossomos) - 1), randint(1, len(individuo.cromossomos) - 1)
        if id1 > id2: id1, id2 = id2, id1
        
        new_ind = Individuo(origens=self.origens, transbordos=self.transbordos, portos=self.portos, clientes=self.clientes)
        for i in range(len(new_ind.cromossomos)):
            new_ind.cromossomos[i].gene_id = individuo.cromossomos[i].gene_id
        new_ind.cromossomos = new_ind.cromossomos[:id1] + new_ind.cromossomos[id2:id1 - 1:-1] + new_ind.cromossomos[id2 + 1:]
        new_ind.give_not_so_random_stuff(cost_matrix=self.cost_matrix)
        if new_ind.get_fitness() <= individuo.get_fitness():
            return new_ind
        return individuo

    def run_evo(
        self,
        fitness_limit: float = -1,
        elitism: int = 0.02,
        tam_population: int = 128,
        generation_limit: int = 1024,
        show_progress: bool = False,
    ) -> Tuple[Individuo, float]:
        population = self.GenPop(tam=tam_population)
        st = time.time()
        nd = None
        best = population[0].get_fitness()
        elitism_size = ceil(tam_population * elitism)
        stagnation = [[0, 0]] * tam_population
        for gen in range(generation_limit):
            population = sorted(
                population,
                key=lambda x: x.get_fitness()
            )
            
            cnt = 0
        
            fmax, favg = float('inf'), 0
            for i in range(tam_population):
                if population[i].get_fitness() - population[0].get_fitness() < 1e-6:
                    cnt += 1
                population[i].rank = i
                fmax = min(fmax, population[i].get_fitness())
                favg += population[i].get_fitness()
            favg = favg/tam_population
            worst = population[-1].get_fitness()
            if show_progress:
                print(f"Generation {gen}, Fitness: min={fmax:.4f}, mean={favg:.4f}, max={population[-1].get_fitness()}, Gap: {population[0].get_fitness()/fitness_limit:.4f}%")
            if population[0].get_fitness() - best < 1e-6:
                best = population[0].get_fitness()
                nd = time.time()
            if abs(best - fitness_limit) < 1e-6:
                break

            next_gen = deepcopy(population[:elitism_size])
            for _ in range(elitism_size, tam_population):
                parents = self.SelectNew(population=population, worst=worst)
                if random.random() < self.crossover_prob(fmax=fmax, favg=favg, fat=parents[0].get_fitness()):
                    child = self.CrossOver(parent_1=parents[0], parent_2=parents[1])
                    if random.random() < self.mutation_prob(fmax=fmax, favg=favg, fat=parents[0].get_fitness()):
                        child = self.Mutation(child)
                    next_gen.append(deepcopy(child))
                else:
                    if random.random() < self.crossover_prob(fmax=fmax, favg=favg, fat=parents[0].get_fitness()):
                        parents[0] = self.Mutation(parents[0])
                    next_gen.append(deepcopy(parents[0]))
            population = deepcopy(next_gen)
        if nd is None:
            nd = time.time()
        return [population[0], nd - st]
            
