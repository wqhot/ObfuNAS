import pickle
import time
from utils import *
from network import *
from setting import *
from mutation import *


def run_evolution_search(victim_spec, cycle=30,
                         population_size=10,
                         tournament_size=5,
                         mutation_rate=1.0, constraint=1e8):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    nasbench.reset_budget_counters()
    victim_val_avg, victim_test_avg = acc_avg(victim_spec) # 调用nasbench.get_metrics_from_spec
    best_scores = [-1000]
    best_valids, best_tests = [victim_val_avg], [victim_test_avg]
    best_history = [victim_spec]
    population = [(victim_val_avg, victim_spec)]  # (validation, spec) tuples
    seed = []

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    i = 0
    while i < population_size:
        # ensure each individual is unique
        while True:
            spec = mutate_spec(victim_spec, victim_spec, mutation_rate)
            flag = check_same(spec, population) # 实际上是check_unique
            if flag == 1:
                break

        print('Generate %d-th new spec' % i)
        # score 是验证集准确率的负数 其值越大 作者认为越好（*）
        score, flops, val_acc, test_acc = fitness(spec)
        if flops < constraint:
            population.append((score, spec))
            i += 1

            # 保证best_scores最后一个是最大的
            if score > best_scores[-1]:
                best_scores.append(score)
                best_history.append(spec)
                best_valids.append(val_acc)
                best_tests.append(test_acc)
            else:
                # 把最大的值复制了一份
                best_scores.append(best_scores[-1])
                best_history.append(best_history[-1])
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
        else:
            rate = flops / constraint
            print("flops / constraint = %f" % rate)

    population.pop(0)
    print('len',len(population))
    print('Iteration:', i)

    # After the population is seeded, proceed with evolving the population.
    i = 0
    while i < cycle:
        # 从 population 中随机选取 tournament_size 个个体
        # C(len(population), tournament_size)
        sample = random_combination(population, tournament_size)
        # print(sample)
        # 从 sample 中选取适应度最高的个体
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        # print(sorted(sample, key=lambda i:i[0]))
        # print(best_spec.original_matrix)
        # 对 best_spec 进行变异
        new_spec = mutate_spec(victim_spec, best_spec, mutation_rate)
        print('The %d-th new spec generated'% i)

        # data = nasbench.query(new_spec)
        # time_spent, _ = nasbench.get_budget_counters()
        # times.append(time_spent)
        score, flops, val_acc, test_acc = fitness(new_spec)

        # In regularized evolution, we kill the oldest individual in the population.
        if flops < constraint:
            population.append((score, spec))
            i += 1
            population.pop(0)

            if score > best_scores[-1]:
                best_scores.append(score)
                best_history.append(new_spec)
                best_valids.append(val_acc)
                best_tests.append(test_acc)
                seed.append(best_spec)
            else:
                best_scores.append(best_scores[-1])
                best_history.append(best_history[-1])
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
                seed.append(victim_spec) # 原始spec

    return best_history, best_valids, best_tests, seed

if __name__=='__main__':
    # nasbench = api.NASBench('nasbench_only108.tfrecord')

    # Query an Inception-like cell from the dataset.
    victim_spec = api.ModelSpec(
        matrix=[[0, 1, 1, 0, 0], # 输入层
                [0, 0, 1, 0, 1], # 1x1卷积
                [0, 0, 0, 1, 0], # 3x3卷积
                [0, 0, 0, 0, 1], # 1x1卷积
                [0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix.
        ops=[INPUT, CONV1X1, CONV3X3, CONV1X1, OUTPUT])


    victim_net = Network(victim_spec)
    input = torch.randn(1, 3, 32, 32)
    victim_flops, _ = profile(victim_net, inputs=(input, ), verbose=False) # 用于计算vectim_net的flops
    print('victim_flops: %f' % victim_flops)


    evolution_data = []
    # search with different flops constraints
    constraint_list = [1e9,8e8,5e8]
    i = 0
    for i in range(len(constraint_list)):
      best_history, best_valids, best_tests, seed = run_evolution_search(victim_spec, cycle=50,
                                                                                     population_size=20,
                                                                                     tournament_size=10,
                                                                                     mutation_rate=1,
                                                                                     constraint=constraint_list[i])

      print('the best mask %s under constrant %s'%(best_history[-1],constraint_list[i]))
      evolution_data.append((best_history, best_valids, best_tests,seed))


    # save results
    time_str = time.strftime("%Y%m%d-%H%M%S")
    with open('data/'+ time_str + 'data.pkl', 'wb') as f:
       pickle.dump(evolution_data, f)
