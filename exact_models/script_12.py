import gurobipy as gp
import numpy as np
import pandas as pd
from math import *

df_ori_dest = pd.read_csv('../dados/origem_porto.csv')
df_ori_trans = pd.read_csv('../dados/origem_transbordo.csv')
df_trans_porto = pd.read_csv('../dados/transbordo_porto.csv')

df_supply = pd.read_csv('../dados/supply.csv')
df_cap_port = pd.read_csv('../dados/cap_porto.csv')
df_cap_trans = pd.read_csv('../dados/cap_transbordo.csv')
df_demand = pd.read_csv('../dados/demand.csv')

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
    supply[i] = df_supply['0'][i]

demand = {}
for i in O:
    demand[i] = df_demand['0'][i - O[0]]

cap_transbordo = {}
for i in K:
    cap_transbordo[i] = df_cap_trans['0'][i - K[0]]

cap_port = {}
for i in M:
    cap_port[i] = df_cap_port['0'][i - M[0]]

cost = {}
for i in N:
    for j in M:
        cost[i, j] = df_ori_dest[str(j - M[0])][i]
    for k in K:
        cost[i, k] = df_ori_trans[str(k - K[0])][i]

for j in M:
    for k in K:
        cost[k, j] = df_trans_porto[str(j - M[0])][k - K[0]]
    for o in O:
        cost[j, o] = 0

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

for j in M:
    for o in O:
        X[j, o] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="X_{}_{}".format(j, o))

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
else:
    for i in N:
        m.addConstr(
            (gp.quicksum(X[i, j] for j in M) + gp.quicksum(X[i, k] for k in K)) <= supply[i]
        )

if oferta_total <= demanda_total:
    for o in O:
        m.addConstr(
            gp.quicksum(X[j, o] for j in M) <= demand[o]
        )
else:
    for o in O:
        m.addConstr(
            gp.quicksum(X[j, o] for j in M) == demand[o]
        )

for j in M:
    m.addConstr(
        (gp.quicksum(X[i, j] for i in N) + gp.quicksum(X[k, j] for k in K)) <= cap_port[j]
    )

for k in K:
    m.addConstr(
        gp.quicksum(X[i, k] for i in N) <= cap_transbordo[k]
    )

for k in K:
    m.addConstr(
        gp.quicksum(X[i, k] for i in N) == gp.quicksum(X[k, j] for j in M)
    )

for j in M:
    m.addConstr(
        gp.quicksum(X[j, o] for o in O) == (gp.quicksum(X[i, j] for i in N) + gp.quicksum(X[k, j] for k in K))
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
