"""
Program to create random cases
Tipo 3:
    demanda <= supply
    fim no porto, sem limite transbordo: 
"""

from math import sqrt, pow, gcd
from random import SystemRandom
import pandas as pd
import numpy as np
from util import GenPoints
from sklearn.linear_model import LinearRegression

def savetocsv(filename, mat):
    arr = np.array([np.array(xi) for xi in mat])
    pd.DataFrame(arr).to_csv("./dados/" + filename + ".csv", index=None)

def create_model(df_path):
    model = LinearRegression()
    df_fit = pd.read_excel(df_path, header=None)
    x, y = np.array(df_fit[0]), np.array(df_fit[1])
    model.fit(X=x[:, np.newaxis], y=y)
    return model
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
    
    model_rodo = create_model('./dados/Frete Rodoviário.xlsx')
    model_ferro = create_model('./dados/Frete Ferroviário.xlsx')
    cost_orig_trans = GenPoints.get_cost(points_orig=points_orig, n=orig,
                                         points_dest=points_transbordo, m=trans, model=model_rodo)
    cost_transbordo_porto = GenPoints.get_cost(points_orig=points_transbordo, n=trans,
                                               points_dest=points_porto, m=port, model=model_ferro)
    cost_orig_porto = GenPoints.get_cost(points_orig=points_orig, n=orig,
                                         points_dest=points_porto, m=port, model=model_rodo)

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
