"""
Program to create random cases
Tipo 1:
    demand <= supply
    fim no cliente
"""

from math import sqrt, pow, gcd
from random import SystemRandom
import pandas as pd
import numpy as np
from util import GenPoints

def savetocsv(filename, mat):
    arr = np.array([np.array(xi) for xi in mat])
    pd.DataFrame(arr).to_csv("./dados/" + filename + ".csv", index=None)

def main():
    orig = int(input("Quantidade origens: "))
    trans = int(input("Quantidade transbordos: "))
    port = int(input("Quantidade portos: "))
    clients = int(input("Quantidade clientes: "))

    points_orig = GenPoints.get_point(lim_x_left=-100, lim_x_right=100,
                            lim_y_down=-100, lim_y_up=100, n=orig)
    
    points_transbordo = GenPoints.get_point(lim_x_left=-250, lim_x_right=250,
                            lim_y_down=-250, lim_y_up=250, n=trans)
    
    points_porto = GenPoints.get_point(lim_x_left=-1000, lim_x_right=1000,
                            lim_y_down=-1000, lim_y_up=1000, n=port)
    
    cost_orig_trans = GenPoints.get_cost(points_orig=points_orig, n=orig, points_dest=points_transbordo, m=trans, tku=0.16)
    cost_transbordo_porto = GenPoints.get_cost(points_orig=points_transbordo, n=trans, points_dest=points_porto, m=port, tku=0.08)
    cost_orig_porto = GenPoints.get_cost(points_orig=points_orig, n=orig, points_dest=points_porto, m=port, tku=0.16)

    savetocsv('origem_transbordo', cost_orig_trans)
    savetocsv('transbordo_porto', cost_transbordo_porto)
    savetocsv('origem_porto', cost_orig_porto)

    demand = [SystemRandom().randint(100, 1000) for _ in range(clients)]
    sum_demand = sum(demand)
    supply = [SystemRandom().randint(100, sum_demand) for _ in range(orig)]
    cap_trans = [SystemRandom().randint(100, 1000) for _ in range(trans)]
    cap_porto = [SystemRandom().randint(100, 1000) for _ in range(port)]

    sum_supply = sum(supply)
    if sum_supply < sum_demand:
        for i in range(orig):
            supply[i] += (sum_demand - sum_supply) // orig
        supply[0] += (sum_demand - sum_supply) % orig
    assert(sum(supply) >= sum_demand)
    aux = np.array(supply)
    pd.DataFrame(aux).to_csv("./dados/supply.csv", index=None)
    
    t = SystemRandom().randint(1, trans)
    for _ in range(t):
        cap_trans[SystemRandom().randint(1, trans) - 1] = sum_demand
    aux = np.array(cap_trans)
    pd.DataFrame(aux).to_csv("./dados/cap_transbordo.csv", index=None)

    t = SystemRandom().randint(1, port)
    for _ in range(t):
        cap_porto[SystemRandom().randint(1, port) - 1] = sum_demand
    aux = np.array(cap_porto)
    pd.DataFrame(aux).to_csv("./dados/cap_porto.csv", index=None)
    
    pd.DataFrame(demand).to_csv("./dados/demand.csv", index=None)


if __name__ == "__main__":
    main()
