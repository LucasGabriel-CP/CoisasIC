import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from random import randint, shuffle, sample, choices, random, seed
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from typing import List, Tuple, Callable, DefaultDict, Dict
from collections import defaultdict
import time
from utils import Point
from GA import Evolution

def closest_distances(cost_matrix, N, M, K) -> Dict[int, list]:
    dists = deepcopy(cost_matrix)
    paths = defaultdict(list())
    for i in N:
        for j in M:
            paths[i] = [j]
    for i in N:
        for j in M:
            for k in K:
                if dists[i][j] > dists[i][k] + dists[k][j]:
                    dists[i][j] = dists[i][k] + dists[k][j]
                    paths[i] = [k, j]
    return paths

def read_stuff() -> list:
    #Read data
    #Read cost matrix
    df_ori_dest = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/origem_porto.csv')
    df_ori_trans = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/origem_transbordo.csv')
    df_trans_porto = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/transbordo_porto.csv')
    
    #Read info about supply and demands
    df_supply = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/supply.csv')
    df_cap_port = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/cap_porto.csv')
    df_cap_trans = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/cap_transbordo.csv')
    df_demand = pd.read_csv('D:/Bibliotecas/Documents/IC/Progs/dados/demand.csv')

    qnt_orig = df_supply.shape[0]
    qnt_trans = df_cap_trans.shape[0]
    qnt_port = df_cap_port.shape[0]
    qnt_cli = df_demand.shape[0]

    N = [i for i in range(qnt_orig)]
    M = [i + qnt_orig + qnt_trans for i in range(qnt_port)]
    K = [i + qnt_orig for i in range(qnt_trans)]
    O = [i + qnt_orig + qnt_trans + qnt_port for i in range(qnt_cli)]

    supply = {}
    soma = 0
    for i in N:
        supply[i] = Point(cap=df_supply['0'][i])
        soma += df_supply['0'][i]
    print(f'supply {soma}')
    demand = {}
    soma = 0
    for i in O:
        demand[i] = Point(cap=df_demand['0'][i - O[0]])
        soma += df_demand['0'][i - O[0]]
    print(f'demand {soma}')

    cap_transbordo = {}
    for i in K:
        cap_transbordo[i] = Point(cap=df_cap_trans['0'][i - K[0]])

    cap_port = {}
    for i in M:
        cap_port[i] = Point(cap=df_cap_port['0'][i - M[0]])

    cost_matrix = defaultdict(dict)
    for i in N:
        for j in M:
            cost_matrix[i][j] = df_ori_dest[str(j - M[0])][i]
        for k in K:
            cost_matrix[i][k] = df_ori_trans[str(k - K[0])][i]

    for j in M:
        for k in K:
            cost_matrix[k][j] = df_trans_porto[str(j - M[0])][k - K[0]]
        for o in O:
            cost_matrix[j][o] = 0
    
    return [supply, soma, cap_transbordo, cap_port, cost_matrix]

def run_ga(cost_matrix, supply: Dict[int, Point], cap_transbordo: Dict[int, Point], cap_port: Dict[int, Point], demand: float, fitness_limit: float) -> list:
    ga = Evolution(origens=supply, transbordos=cap_transbordo, portos=cap_port, demand=demand, cost_matrix=cost_matrix)
    # breakpoint()
    [best, t] = ga.run_evo(
                        generation_limit=10000, tam_population=144,
                        elitism=.05, fitness_limit=fitness_limit,
                        show_progress=True
                    )
    score = best.get_fitness()
    graph = defaultdict(lambda: np.zeros(2))
    for i in best.cromossomos:
        for u, val in i.adjlist.items():
            for v, val2 in val.items():
                graph[u, v] += val2
    return [graph, score]


def main() -> None:
    t = int( time.time() * 1000.0 )
    seed( ((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24)   )
    [supply, demand, cap_transbordo, cap_port, cost_matrix] = read_stuff()

    media, best_score = 0, float('inf')
    fl = 1050097.9903317615
    graph = defaultdict(lambda: np.zeros(2))

    all_scores = []
    mean_gap = 0
    for _ in range(10):
        [ga_graph, ga_score] = run_ga(cost_matrix=cost_matrix, supply=supply, cap_transbordo=cap_transbordo,
                                    cap_port=cap_port,demand=demand, fitness_limit=fl)
        if ga_score < best_score:
            best_score = ga_score
            graph = ga_graph
        gap = (ga_score - fl) / ((ga_score + fl) / 2) * 100
        mean_gap += gap
        print(f'genetic algorithm fitness {ga_score} with {gap:.4f}% gap')
        all_scores.append(ga_score)
        media += ga_score
    print(f'resultado medio: {media/10}\ngap medio: {mean_gap/10}')
    
    df = pd.DataFrame()
    df['scores'] = all_scores
    df.to_excel('./result.xlsx', index=False)

    send = defaultdict(float)
    file_ans = open('./graph.txt', 'w')
    for key, value in graph.items():
        u, v = key
        f = ''
        if u in cap_port:
            f = 'port'
        elif u in cap_transbordo:
            f = 'transhipment'
        elif u in supply:
            f = 'supply'
        s = ''
        if v in cap_port:
            s = 'port'
        elif v in cap_transbordo:
            s = 'transhipment'
        elif u in supply:
            s = 'supply'
        send[u] += value[1]
        file_ans.write(f'{f} {u} send {value[1]} to {s} {v}\n')

    for key, value in send.items():
        tot = 0
        what = ''
        if key in cap_port:
            tot = cap_port[key].cap
            what = 'port'
        if key in cap_transbordo:
            tot = cap_transbordo[key].cap
            what = 'transhipment'
        if key in supply:
            tot = supply[key].cap
            what = 'supply'
        file_ans.write(f'{what} {key} send: {value} from total of {tot}\n')
    file_ans.close()


if __name__ == "__main__":
    main()