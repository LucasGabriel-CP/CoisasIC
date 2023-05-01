import numpy as np
import matplotlib.pyplot as plt
from random import randint
from copy import deepcopy
from math import sqrt, cos, acos, ceil
from functools import partial
from random import sample, choices, random
from typing import List, Tuple, Callable
import time

def proper_round(num, dec=0)->float:
    return round(num, dec)

def DistanceBetweenPoints(Pi: List[float], Pf: List[float]) -> int:
  return int(proper_round(sqrt((Pf[0] - Pi[0]) * (Pf[0] - Pi[0]) + (Pf[1] - Pi[1]) * (Pf[1] - Pi[1]))))

def showGraph(Vetor: List[int], pontos: List[Tuple[float, float]], fit: int, showCord: bool = False) -> None: #Função para plotar os gráficos
  #Distruibção dos pontos em dois vetores X e Y
  X, Y = [], []
  for i in Vetor:
    X.append(pontos[i][0])
    Y.append(pontos[i][1])
    if showCord:
      print(f'X: {X[-1]:3} - Y: {Y[-1]}')
  
  AuxX, AuxY = [], []
  AuxX.append(X[0])
  AuxX = AuxX + X[1:] + AuxX
  AuxY.append(Y[0])
  AuxY = AuxY + Y[1:] + AuxY

  plt.figure(figsize=(12,12))
  plt.plot(AuxX, AuxY, color='black', alpha=1)
  plt.scatter(X, Y, color='red', s=25)#Plot dos pontos
  plt.grid()
  plt.title('Distância: ' + str(fit),
            fontsize=12,
            color='red',
            loc='right')
  plt.show()

def GxT(X: List[int], Y: List[float]) -> None:
  plt.figure(figsize=(25,7))
  plt.plot(X, Y)
  plt.grid()
  plt.xlabel('Generation')
  plt.ylabel('Time')
  plt.title('total time: ' + str(sum(Y)),
            fontsize=12,
            color='red',
            loc='right')
  plt.show()

def GxE(X: List[int], Y: List[int]) -> None:
  plt.figure(figsize=(25,7))
  plt.plot(X, Y)
  plt.grid()
  plt.xlabel('Generation')
  plt.ylabel('Distance')
  plt.show()

Pts = List[List[int]]
Individuo = List[int]
Populacao = List[Individuo]
FitnessFunc = Callable[[Individuo], int]
PopulationFunc = Callable[[], Populacao]
SelectFunc = Callable[[Populacao, Pts, FitnessFunc], Tuple[Individuo, Individuo]]
CrossOverFunc = Callable[[Individuo, Individuo], Tuple[Individuo, Individuo]]
MutationFunc = Callable[[Individuo], Individuo]

def GenerateInd(pontos: List[int]) -> Individuo:
  return sample(pontos, len(pontos))
def GenPop(pontos: Pts, Tam: int) -> Populacao:
  return [GenerateInd([i for i in range(len(pontos))]) for _ in range(Tam)]
def Fitness(individuo: Individuo, pontos: Pts) -> int:
  if len(individuo) != len(pontos):
    raise ValueError('Tamanhos diferentes')
  
  Dist = 0
  for i in range(1, len(individuo)):
    Dist += pontos[individuo[i-1]][individuo[i]]
    
  return Dist + pontos[individuo[0]][individuo[-1]]

def SelectNew(population: Populacao, worst: int, pontos: Pts, fitness_func: FitnessFunc) -> Populacao:
  return choices(
      population=population,
      weights=[worst-fitness_func(individuo=individuo) for individuo in population],
      k=2
  )

def CrossOver(a: Individuo, b: Individuo) -> Tuple[Individuo, Individuo]:
  if len(a) != len(b):
    raise ValueError('Tamanhos diferentes')
  N = len(a)
  p1, p2 = randint(0, N-1), randint(0, N-1)
  while p1 == p2: p2 = randint(0, N-1)
  NewInd1 = a[p1:p2]
  NewInd2 = b[p1:p2]

  for i in range(N):
    P = b[i]
    if P not in NewInd1:
      NewInd1.append(P)
  for i in range(N):
    P = a[i]
    if P not in NewInd2:
      NewInd2.append(P)

  return NewInd1, NewInd2

def Mutation(individuo: Individuo, pontos: Pts) -> Individuo:
  improve = True
  while improve:
    N = len(individuo)
    min_change, DistAux = 0, Fitness(individuo, pontos)

    improve = False
    for i in range(N - 2):
      for j in range(i + 2, N-1):
        change = (pontos[individuo[i]][individuo[j]] + pontos[individuo[i+1]][individuo[j+1]]
                - pontos[individuo[i]][individuo[i+1]] - pontos[individuo[j]][individuo[j+1]])
        if change < min_change:
          min_change = change
          min_i, min_j = i, j
    if min_change < 0:
      individuo[min_i+1:min_j+1] = individuo[min_i+1:min_j+1][::-1]
    if DistAux > Fitness(individuo, pontos):
      improve = True
  return individuo

def Mutation1(individuo: Individuo, pontos: Pts) -> Individuo:
  N = len(individuo)
  p1, p2 = randint(0, N-1), randint(0, N-1)
  while p1 == p2: p2 = randint(0, N-1) # if p1 == p2: p2 += 1

  if p1 > p2: p1, p2 = p2, p1
  
  return individuo[:p1] + individuo[p1:p2][::-1] + individuo[p2:]

def run_evo(
    populate_func: PopulationFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    Dists: Pts,
    selection_func: SelectFunc = SelectNew,
    crossover_func: CrossOverFunc = CrossOver,
    mutation_func: MutationFunc = Mutation,
    generation_limit: int = 100,
    show_progress: bool = False,
    croosover_point: int = 0.5,
    mutation_point: int = 0.35
) -> Tuple[Populacao, int, List[int], List[int], List[int]]:
  population = populate_func()
  start, stop, VetT, VetG, results = 0, 0, [], [], []
  for i in range(generation_limit):
    population = sorted(
        population,
        key=lambda individuo: fitness_func(individuo=individuo)
    )

    if show_progress:
      print(f'Generation: {i}. Result: {fitness_func(individuo=population[0])}. Time: {stop-start}')
    VetT.append(stop-start)
    VetG.append(i)
    results.append(fitness_func(individuo=population[0]))
    if fitness_func(individuo=population[0]) <= fitness_limit:
      break

    start = time.time()
    next_gen = population[:2]
    worst = fitness_func(population[-1])
    for _ in range(int(len(population)/2)-1):
      parents = selection_func(population=population, worst=worst, pontos=Dists, fitness_func=fitness_func)
      if random() > croosover_point:
        sobrevivente_a, sobrevivente_b = crossover_func(a=parents[0], b=parents[1])
        if random() > mutation_point:
          sobrevivente_a = mutation_func(individuo=sobrevivente_a, pontos=Dists)
          sobrevivente_b = mutation_func(individuo=sobrevivente_b, pontos=Dists)
        next_gen += [sobrevivente_a, sobrevivente_b]
      else:
        if random() > mutation_point:
          parents[0] = mutation_func(individuo=parents[0], pontos=Dists)
          parents[1] = mutation_func(individuo=parents[1], pontos=Dists)
        next_gen += [parents[0], parents[1]]
    population = deepcopy(next_gen)

    stop = time.time()

  population = sorted(
        population,
        key=lambda individuo: fitness_func(individuo=individuo)
    )

  return population, i, VetT, VetG, results

Pt = []
file = open("./kroA100.tsp")
for line in file:
  if line == 'EOF\n': break
  if line == '\n': continue
  aux = line.split()
  pt = [float(aux[1]), float(aux[2])]
  if pt not in Pt:
    Pt.append(pt)

Distances = []
for i in range(len(Pt)):
  Distance = []
  for j in range(len(Pt)):
    if i == j:
      Distance.append(0)
    else:
      Distance.append(DistanceBetweenPoints(Pt[i], Pt[j]))
  Distances.append(deepcopy(Distance))

population, generation, VetT, VetG, VetR = run_evo(
    populate_func=partial(
        GenPop, pontos=Distances, Tam=100
    ),
    fitness_func=partial(
        Fitness, pontos=Distances
    ),
    Dists=Distances,
    mutation_func=Mutation,
    fitness_limit=21282,
    generation_limit=425,
    show_progress=True
)

showGraph(Vetor=population[0], pontos=Pt, fit=Fitness(population[0], Distances))

GxT(VetG, VetT)