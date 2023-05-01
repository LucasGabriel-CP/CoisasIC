"""
Program to create random cases
Tipo 3:
    demanda <= supply
    fim no porto, sem limite transbordo: 
"""

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

    points_orig = GenPoints.get_point(lim_x_left=-100, lim_x_right=100,
                            lim_y_down=-100, lim_y_up=100, n=orig)
    points_transbordo = GenPoints.get_point(lim_x_left=-250, lim_x_right=250,
                            lim_y_down=-250, lim_y_up=250, n=trans)
    points_porto = GenPoints.get_point(lim_x_left=-1000, lim_x_right=1000,
                            lim_y_down=-1000, lim_y_up=1000, n=port)
    
    cost_orig_porto = GenPoints.get_cost(points_orig=points_orig, n=orig,
                                         points_dest=points_porto, m=port, tku=0.16)
    cost_orig_trans = GenPoints.get_cost(points_orig=points_orig, n=orig,
                                         points_dest=points_transbordo, m=trans, tku=0.16)
    cost_transbordo_porto = GenPoints.get_cost(points_orig=points_transbordo, n=trans,
                                                points_dest=points_porto, m=port, tku=0.08)

    savetocsv('origem_transbordo', cost_orig_trans)
    savetocsv('transbordo_porto', cost_transbordo_porto)
    savetocsv('origem_porto', cost_orig_porto)

    demmand = [SystemRandom().randint(100, 1000) for _ in range(port)]
    sum_demmand = sum(demmand)
    supply = [SystemRandom().randint(100, sum_demmand) for _ in range(orig)]
    sum_supply = sum(supply)

    if sum_demmand > sum_supply:
        for i in range(orig):
            supply[i] += abs(sum_supply - sum_demmand) // orig
        supply[0] += abs(sum_supply - sum_demmand) % orig
    aux = np.array(supply)
    pd.DataFrame(aux).to_csv("./dados/supply.csv", index=None)
    
    pd.DataFrame(demmand).to_csv("./dados/demand.csv", index=None)


if __name__ == "__main__":
    main()
