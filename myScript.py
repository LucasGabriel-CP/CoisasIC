import exact_models.gurobi_12 as gb
import genetic_algo.gaTester as ga
import gera_casos_tipo1 as gerador
import time
from random import seed

t = int(time.time() * 1000.0 )
seed( ((t & 0xff000000) >> 24) +
            ((t & 0x00ff0000) >>  8) +
            ((t & 0x0000ff00) <<  8) +
            ((t & 0x000000ff) << 24)   )


file = open('instances.txt', 'r')
demandas, ofertas, ans = [], [], []
pops = [377, 377, 377, 377, 75, 75, 144, 144, 75]
elts = [.1,   .1,  .1,  .1,  .1,  .1, .05, .05, .1]
done = [True, True, True, True, True, True, False, False, False]
for i in range(9):
    n, m, p = map(int, list(file.readline().split()))
    if done[i]: continue
    gerador.create_instance(instance_id=i, orig=n, trans=m, port=p)
    [solver_result, a, b] = gb.run_gurobi()
    print(f'Melhor: {solver_result}\ndemanda: {a}\noferta: {b}')
    demandas.append(a)
    ofertas.append(b)
    ans.append(solver_result)

    ga.run_tests(solver_result, i, pops[i], elts[i])

print(f'demandas: {demandas}')
print(f'ofertas: {ofertas}')
print(ans)