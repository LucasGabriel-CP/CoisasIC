import gurobipy as gp

mdl = gp.Model("VRP")

'''Sets and parameters'''

N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
V = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = [1]
A = [(i,j) for i in V for j in V if i!=j]
q = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
C = {1:20}
M = {1:100}

'''cost matrix'''

c = {(0,1): 1, (0, 2): 2, (0, 3): 4, (0, 4): 5, (0, 5): 6, (0, 6): 8, (0, 7): 9, (0, 8): 11, (0, 9): 12, (0, 10): 13,
    (1, 0): 1, (1, 2): 1, (1, 3): 3, (1, 4): 4, (1, 5): 6, (1, 6): 7, (1, 7): 8, (1, 8): 10, (1, 9): 11, (1, 10): 13,
    (2, 0): 2, (2, 1): 1, (2, 3): 1, (2, 4): 3, (2, 5): 4, (2, 6): 6, (2, 7): 7, (2, 8): 8, (2, 9): 10, (2, 10): 11,
    (3, 0): 4, (3, 1): 3, (3, 2): 1, (3, 4): 1, (3, 5): 3, (3, 6): 4, (3, 7): 6, (3, 8): 7, (3, 9): 8, (3, 10): 10,
    (4, 0): 5, (4, 1): 4, (4, 2): 3, (4, 3): 1, (4, 5): 1, (4, 6): 3, (4, 7): 4, (4, 8): 6, (4, 9): 7, (4, 10): 8,
    (5, 0): 6, (5, 1): 6, (5, 2): 4, (5, 3): 3, (5, 4): 1,(5, 6): 1, (5, 7): 3, (5, 8): 4, (5, 9): 6, (5, 10): 7,
    (6, 0): 8, (6, 1): 7, (6, 2): 6, (6, 3): 4, (6, 4): 3, (6, 5): 1, (6, 7): 1, (6, 8): 3, (6, 9): 4, (6, 10): 6,
    (7, 0): 9, (7, 1): 8, (7, 2): 7, (7, 3): 6, (7, 4): 4, (7, 5): 3, (7, 6): 1,(7, 8): 1, (7, 9): 3, (7, 10): 4,
    (8, 0): 11, (8, 1): 10, (8, 2): 8, (8, 3): 7, (8, 4): 6, (8, 5): 4, (8, 6): 3, (8, 7): 1, (8, 9): 1, (8, 10): 3,
    (9, 0): 12, (9, 1): 11, (9, 2): 10, (9, 3): 8, (9, 4): 7, (9, 5): 6, (9, 6): 4, (9, 7): 3, (9, 8): 1, (9, 10): 1,
    (10, 0): 13, (10, 1): 13, (10, 2): 11, (10, 3): 10, (10, 4): 8, (10, 5): 7, (10, 6): 6, (10, 7): 4, (10, 8): 3, (10, 9): 1}

'''Decision Variable'''
pairs = [(i,j,k) for i in V for j in V for k in K]
x = mdl.addVars(pairs, vtype=gp.GRB.BINARY, name ="x")
mdl.modelSense = gp.GRB.MINIMIZE

'''Constraints'''

mdl.addConstrs((gp.quicksum(x[i,j,k] for i in V) - gp.quicksum(x[j,i,k] for i in V)) == 0 for j in V for k in K)
mdl.addConstrs(gp.quicksum(x[i,j,k] for k in K for i in V if i != j) == 1 for j in N)
mdl.addConstrs(x[0,j,k] == 1 for j in V for k in K)
mdl.addConstrs(gp.quicksum(x[0,j,k] for j in N ) <= M[k] for k in K)
mdl.addConstrs(gp.quicksum(x[i,j,k]*q[i] for i in V for j in N if j != i) <= C[k] for k in K)

mdl.addConstrs(x[i,i,k] == 0 for i in V for k in K)

'''Objective Function'''
mdl.setObjective(gp.quicksum(x[i,j,k]*c[(i,j)] for (i,j) in A for k in K))

'''Solve'''

mdl.optimize()
