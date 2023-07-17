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
from GA import Evolution

def main() -> None:
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
    for i in N:
        supply[i] = Point(cap=df_supply['0'][i])

    demand = {}
    for i in O:
        demand[i] = Point(cap=df_demand['0'][i - O[0]])

    cap_transbordo = {}
    for i in K:
        cap_transbordo[i] = Point(cap=df_cap_trans['0'][i - K[0]])

    cap_port = {}
    for i in M:
        cap_port[i] = Point(cap=df_cap_port['0'][i - M[0]])

    cost_matrix = {}
    for i in N:
        for j in M:
            cost_matrix[i, j] = df_ori_dest[str(j - M[0])][i]
        for k in K:
            cost_matrix[i, k] = df_ori_trans[str(k - K[0])][i]

    for j in M:
        for k in K:
            cost_matrix[k, j] = df_trans_porto[str(j - M[0])][k - K[0]]
        for o in O:
            cost_matrix[j, o] = 0

    ga = Evolution(origens=supply, transbordos=cap_transbordo, portos=cap_port, clientes=demand, cost_matrix=cost_matrix)
    ga.run_evo()
    

if __name__ == "__main__":
    main()