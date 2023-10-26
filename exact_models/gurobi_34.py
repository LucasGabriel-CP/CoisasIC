import gurobipy as gp
import numpy as np
import pandas as pd
from math import *

df_ori_dest = pd.read_csv('../dados/origem_porto.csv')
df_ori_trans = pd.read_csv('../dados/origem_transbordo.csv')
df_trans_porto = pd.read_csv('../dados/transbordo_porto.csv')

df_supply = pd.read_csv('../dados/supply.csv')
df_demand = pd.read_csv('../dados/demand.csv')

qnt_orig = df_supply.shape[0]
qnt_trans = df_trans_porto.shape[0]
qnt_port = df_demand.shape[0]

N = [i for i in range(qnt_orig)]
M = [i + qnt_orig + qnt_trans for i in range(qnt_port)]
K = [i + qnt_orig for i in range(qnt_trans)]

supply = {}
for i in N:
    supply[i] = df_supply['0'][i]

demand = {}
for i in M:
    demand[i] = df_demand['0'][i - M[0]]

cost = {}
for i in N:
    for j in M:
        cost[i, j] = df_ori_dest[str(j - M[0])][i]
    for k in K:
        cost[i, k] = df_ori_trans[str(k - K[0])][i]

for j in M:
    for k in K:
        cost[k, j] = df_trans_porto[str(j - M[0])][k - K[0]]

oferta_total = 0
for i in supply:
    oferta_total += supply[i]
demanda_total = 0
for i in demand:
    demanda_total += demand[i]
print(oferta_total, demanda_total)

m = gp.Model("probrema")

X = {}
for i in N:
    for j in M:
        X[i, j] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="X_{}_{}".format(i, j))

for i in N:
    for k in K:
        X[i, k] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="X_{}_{}".format(i, k))
                
for j in M:
    for k in K:
        X[k, j] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="X_{}_{}".format(k, j))

m.setObjective(
    gp.quicksum((X[i, j] * cost[i, j]) for i in N for j in M) + \
    gp.quicksum((X[i, k] * cost[i, k]) for i in N for k in K) + \
    gp.quicksum((X[k, j] * cost[k, j]) for j in M for k in K),
    sense=gp.GRB.MINIMIZE
)

if oferta_total <= demanda_total:
    for i in N:
        m.addConstr(
            (gp.quicksum(X[i, j] for j in M) + gp.quicksum(X[i, k] for k in K)) == supply[i]
        )
    for j in M:
        m.addConstr(
            (gp.quicksum(X[i, j] for i in N) + gp.quicksum(X[k, j] for k in K)) <= demand[j]
        )

else:
    for i in N:
        m.addConstr(
            (gp.quicksum(X[i, j] for j in M) + gp.quicksum(X[i, k] for k in K)) <= supply[i]
        )
    for j in M:
        m.addConstr(
            (gp.quicksum(X[i, j] for i in N) + gp.quicksum(X[k, j] for k in K)) == demand[j]
        )


for k in K:
    m.addConstr(
        gp.quicksum(X[i, k] for i in N) == gp.quicksum(X[k, j] for j in M)
    )

rest = m.addConstrs(
    gp.quicksum(X[i, j] for i in N) >= 0 for j in M
)

rest = m.addConstrs(
    gp.quicksum(X[i, k] for i in N) >= 0 for k in K
)

rest = m.addConstrs(
    gp.quicksum(X[k, j] for k in K) >= 0 for j in M
)

m.update()
m.optimize()


