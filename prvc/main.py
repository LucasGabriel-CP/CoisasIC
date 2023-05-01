import numpy as np
import matplotlib.pyplot as plt
from random import randint
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from random import sample, choices, random
from typing import List, Tuple, Callable
import time

def proper_round(num, dec=0) -> float:
    return round(num, dec)

class Point:
    def __init__(self, x, y, demand = 0) -> None:
        self.x = x
        self.y = y
        self.demand = demand

    def get_distance(self, other) -> int:
        d = sqrt((other.x -  self.x) * (other.x -  self.x) +
                    (other.y -  self.y) * (other.y -  self.y))
        return int(proper_round(d))

    def __str__(self) -> str:
        return 'x: ' + str(self.x) + '\ny: ' + str(self.y) + '\ndemand: ' + str(self.demand)

class Vehicle:
    def __init__(self, Q: int, depot_id: int) -> None:
       self.capacity = Q
       self.n = 0
       self.depot_id = depot_id
       self.distance = 0
       self.route = []

    def set_distance(self, cost_matrix: List[List[float]]) -> None:
        self.distance = 0
        for i in range(1, self.n):
            #d = Clients[self.route[i]].get_distance(Clients[self.route[i - 1]])
            #breakpoint()
            self.distance += cost_matrix[self.route[i]][self.route[i-1]]
        self.distance += cost_matrix[self.route[-1]][0]
        #breakpoint()

    def set_route2(self, Clients: List[Point], cost_matrix: List[List[float]], ids: List[int], vis: List[bool], n: int) -> int:
        i = 0

        if ids[i] != self.depot_id:
            self.route.append(self.depot_id)
            self.n += 1

        while i < n and self.capacity - Clients[ids[i]].demand >= 0:
            if not vis[ids[i]]:
                self.capacity -= Clients[ids[i]].demand
                self.route.append(ids[i])
                vis[ids[i]] = True
                self.n += 1
            i += 1
        self.set_distance(cost_matrix=cost_matrix)

    def set_route(self, Clients: List[Point], cost_matrix: List[List[float]], ids: List[int], n: int, start: int) -> int:
        i = start
        self.route.append(self.depot_id)
        self.n += 1

        while i < n and self.capacity - Clients[ids[i]].demand >= 0:
            self.capacity -= Clients[ids[i]].demand
            self.route.append(ids[i])
            i += 1
            self.n += 1
        self.set_distance(cost_matrix=cost_matrix)

        return i

    def appy_2opt(self, cost_matrix: List[List[float]]):
        improve = True
        while improve:
            min_change, bigger_dist = 0, self.distance
            
            improve = False
            for i in range(self.n - 2):
                for j in range(i + 2, self.n - 1):
                    change = (cost_matrix[self.route[i]][self.route[j]] + cost_matrix[self.route[i + 1]][self.route[j + 1]]
                            - cost_matrix[self.route[i]][self.route[i + 1]] - cost_matrix[self.route[j]][self.route[j + 1]])
                    if change < min_change:
                        min_change = change
                        min_i, min_j = i, j
            if min_change < 0:
                self.route[min_i + 1:min_j+1] = self.route[min_i + 1:min_j+1][::-1]
            self.set_distance(cost_matrix=cost_matrix)
            if bigger_dist > self.distance:
                improve = True

    def __str__(self) -> str:
        return str(self.route) + '\n' + str(self.distance)

Pts = List[Point]
Individuo = List[Vehicle]
Populacao = List[Individuo]
FitnessFunc = Callable[[Individuo], int]
PopulationFunc = Callable[[Pts, int, int], Populacao]
SelectFunc = Callable[[Populacao, Pts, FitnessFunc], Tuple[Individuo, Individuo]]
CrossOverFunc = Callable[[Individuo, Individuo], Tuple[Individuo, Individuo]]
MutationFunc = Callable[[Individuo], Individuo]

'''
    gerar sample aleatório
    mandar inserir até encher
    retornar quais faltam
'''
def GenerateSample(Clients: List[Point], cost_matrix: List[List[float]], idx: List[int], Q: int, depot_id: int) -> Individuo:
    n = len(idx)
    sp = sample(idx, randint(1, n))
    vis = [False] * (n + 1)
    new_ind = []
    while True:
        vehicle = Vehicle(Q=Q, depot_id=depot_id)
        vehicle.set_route(Clients=Clients, cost_matrix=cost_matrix, ids=sp, vis=vis, n=len(sp))
        if vehicle.n:
            new_ind.append(vehicle)
        
    return

def GenerateInd(Clients: List[Point], cost_matrix: List[List[float]], idx: List[int], Q: int, depot_id: int) -> Individuo:
    n = len(idx)
    P = sample(idx, n)
    new_ind = []
    i = 0
    while i < n:
        vehicle = Vehicle(Q=Q, depot_id=depot_id)
        i = vehicle.set_route(Clients=Clients, cost_matrix=cost_matrix, ids=P, n=n, start = i)
        new_ind.append(vehicle)



    return new_ind

def GenPop(Clients: Pts, cost_matrix: List[List[float]], Tam: int, Q: int, depot_id: int) -> Populacao:
    return [GenerateInd(Clients, cost_matrix, [i for i in range(1, len(Clients))], Q, depot_id) for _ in range(Tam)]

def Fitness(individuo: Individuo) -> int:
    sum = 0
    for vehi in individuo:
        sum += vehi.distance
    return sum

def SelectNew(population: Populacao, worst: int, fitness_func: FitnessFunc) -> Populacao:
    return choices(
        population=population,
        weights=[1 + worst-fitness_func(individuo=individuo) for individuo in population],
        k=2
    )

def CrossOver(parent_1: Individuo, parent_2: Individuo, Clients: List[Point],
                cost_matrix: List[List[float]], Q: int, depot_id: int) -> Individuo:
    sorted_parent_1 = sorted(
        parent_1, key=lambda x: x.distance
    )

    #pegar os M/2 caminhos mais curtos do primeiro pai
    child_routes = sorted_parent_1[:(len(sorted_parent_1) + 1)//2]

    has = [0 for _ in range(len(Clients))]
    for veh in child_routes:
        for p in veh.route:
            has[p] = 1
    Children = deepcopy(child_routes)

    #Adicionar as rotas vindo do pai
    new_route = []
    for veh in parent_2:
        for p in veh.route:
            if not has[p]:
                new_route.append(p)
                has[p] = 1
    #breakpoint()
    
    i = 0
    while i < len(new_route):
        new_vehicle = Vehicle(Q=Q, depot_id=depot_id)
        i = new_vehicle.set_route(Clients=Clients, cost_matrix=cost_matrix, ids=new_route, n=len(new_route), start=i)
        Children.append(new_vehicle)
    
    #breakpoint()
    for i in has:
        assert i == 1, 'Invalid route'
    
    return Children

def Mutation(individuo: Individuo, Clients: List[Point],  cost_matrix: List[List[float]]):
    for u in range(len(individuo) - 1):
        v = randint(u + 1, len(individuo) - 1)

        id1, id2 = randint(0, individuo[u].n - 1), randint(0, individuo[v].n - 1)
        c1, c2 = individuo[u].route[id1], individuo[v].route[id2]
        if c1 == individuo[u].depot_id or c2 == individuo[v].depot_id:
            continue
        if (individuo[u].capacity + Clients[c1].demand - Clients[c2].demand >= 0
                and individuo[v].capacity + Clients[c2].demand - Clients[c1].demand >= 0):
            individuo[u].route[id1], individuo[v].route[id2] = c2, c1
            individuo[u].capacity += (Clients[c1].demand - Clients[c2].demand)
            individuo[v].capacity += (Clients[c2].demand - Clients[c1].demand)
            individuo[u].set_distance(cost_matrix=cost_matrix)
            individuo[v].set_distance(cost_matrix=cost_matrix)
    
    has = [0 for _ in range(len(Clients))]
    for i in range(len(individuo)):
        individuo[i].appy_2opt(cost_matrix=cost_matrix)
        for i in individuo[i].route:
            has[i] = 1
    
    #breakpoint()
    for i in has:
        assert i == 1, 'Invalid route'
    return individuo

def run_evo(
    populate_func: PopulationFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    Q: int,
    depot_id: int,
    Clients: List[Point],
    cost_matrix: List[List[float]],
    selection_func: SelectFunc = SelectNew,
    crossover_func: CrossOverFunc = CrossOver,
    mutation_func: MutationFunc = Mutation,
    generation_limit: int = 100,
    show_progress: bool = False,
    croosover_point: int = 0.35,
    mutation_point: int = 0.35
) -> Tuple[Populacao, int, List[int], List[int], List[int]]:
    population = populate_func(cost_matrix=cost_matrix)
    start, stop, results = 0, 0, []
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda individuo: fitness_func(individuo=individuo)
        )

        if show_progress:
            print(f'Generation: {i}. Result: {fitness_func(individuo=population[0])}. Time: {stop-start}')
        results.append(fitness_func(individuo=population[0]))

        if fitness_func(individuo=population[0]) <= fitness_limit:
            break

        start = time.time()
        next_gen = deepcopy(population[:2])
        worst = fitness_func(population[-1])
        for _ in range(int(len(population)/2)-1):
            parents = selection_func(population=population, worst=worst, fitness_func=fitness_func)
            if random() > croosover_point:
                child = crossover_func(parent_1=parents[i & 1], parent_2=parents[not (i & 1)],
                                        cost_matrix=cost_matrix, Q=Q, depot_id=depot_id, Clients=Clients)
                if random() > mutation_point:
                    child = mutation_func(individuo=child, Clients=Clients, cost_matrix=cost_matrix)
                next_gen += [child, parents[0]]
            else:
                if random() > mutation_point:
                    parents[0] = mutation_func(individuo=parents[0], Clients=Clients, cost_matrix=cost_matrix)
                    parents[1] = mutation_func(individuo=parents[1], Clients=Clients, cost_matrix=cost_matrix)
                next_gen += deepcopy([parents[0], parents[1]])

        population = deepcopy(next_gen)

        stop = time.time()

    population = sorted(
        population,
        key=lambda individuo: fitness_func(individuo=individuo)
    )

    return population, i, results

def plot_graph(result: Individuo, Clients: List[Point], depot_id: Point):
    for vehi in result:
        x, y = [], []
        for i in vehi.route:
            x.append(Clients[i].x)
            y.append(Clients[i].y)
        x.append(Clients[vehi.route[0]].x)
        y.append(Clients[vehi.route[0]].y)
        plt.plot(x, y)
    #plt.scatter(depot_id, depot_id)
    plt.show()

def read_file(name: str):
    which = 0
    points = []
    Q, n, cnt = 0, 0, 0
    depot_id = -1
    file = open(name)
    for line in file:
        if line == 'EOF':
            break
        line = line.split()
        print(line)
        if which == 1:
            points.append(Point(int(line[1]), int(line[2])))
            cnt += 1
            if cnt == n:
                which, cnt = 0, 0
        elif which == 2:
            points[int(line[0]) - 1].demand = int(line[1])
            cnt += 1
            if cnt == n:
                which, cnt = 0, 0
        elif which == 3:
            depot_id = int(line[0]) - 1
            #points.remove(points[int(line[0]) - 1])
            break
        elif line[0] == 'DIMENSION':
            n = int(line[2])
        elif line[0] == 'CAPACITY':
            Q = int(line[2])
        elif line[0] == 'NODE_COORD_SECTION':
            which = 1
        elif line[0] == 'DEMAND_SECTION':
            which = 2
        elif line[0] == 'DEPOT_SECTION':
            which = 3
    
    return points, Q, depot_id

def main():
    Clients, Q, depot_id = read_file('A-n32-k5.vrp.txt')
    cost_matrix = []
    for i in range(len(Clients)):
        aux = []
        for j in range(len(Clients)):
            if i == j:
                aux.append(0)
            else:
                aux.append(Clients[i].get_distance(Clients[j]))
        #     print(f'{i+1, j+1}: {int(aux[-1])}, ', end='')
        # print()
        cost_matrix.append(deepcopy(aux))
    breakpoint()

    population, generation, VetR = run_evo(
        populate_func=partial(
            GenPop, Clients=Clients, Tam=256, Q=Q, depot_id=depot_id
        ),
        fitness_func=Fitness,
        mutation_func=Mutation,
        Clients=Clients,
        cost_matrix=cost_matrix,
        fitness_limit=-1,
        generation_limit=10024,
        Q=Q,
        depot_id=depot_id,
        show_progress=True
    )
    print(f'Resultado: {Fitness(population[0])}')
    breakpoint()
    plot_graph(result=population[0], Clients=Clients, depot_id=depot_id)

if __name__ == '__main__':
    main()