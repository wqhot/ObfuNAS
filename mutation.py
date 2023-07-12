from setting import *
import copy
import numpy as np
import random

def mutate_spec(victim_spec, old_spec, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)
        # victim_spec = copy.deepcopy(old_spec)
        NUM_VERTICES = new_matrix.shape[0]
        idle_VERTICES = MAX_VERTICES - NUM_VERTICES # MAX_VERTICES = 7 剩余可增长的空闲结点数
        seq_list = np.arange(NUM_VERTICES).tolist()
        para_list = np.arange(NUM_VERTICES - 1).tolist()  # avoid output

        # In expectation, one op is resampled.
        # Kernel widening  OP_SPOTS = NUM_VERTICES - 2，即扣除了输入和输出的层数 每一层都有可能变异，共享变异概率
        op_mutation_prob = mutation_rate / OP_SPOTS 
        for ind in range(1, NUM_VERTICES - 1):
            if new_ops[ind] == CONV1X1: # 只有CONV1X1 -> CONV3X3 这一种变异方向
                if random.random() < op_mutation_prob:
                    new_ops[ind] = CONV3X3

        # 如果当前结点数未达到MAX_VERTICES(7)，且边数未达到MAX_EDGES(9)
        # Layer deepening
        if new_matrix.sum() < MAX_EDGES and idle_VERTICES > 0:
            # random select 2 ops and add another op between them

            [x, y] = sorted(random.sample(seq_list, 2)) # 随机从 seq_list 中选取两个结点
            if new_matrix[x, y] == 1 and [x, y] != [0, seq_list[-1]]: # x y 层之间存在连接关系， 且不是输入和输出层之间的连接
                op_add_prob = mutation_rate / 5  # 3 options: parallel adding, sequential adding, and no change 5是怎么来的
                # add one sequential op
                if random.random() < op_add_prob: # 增加一个顺序连接层
                    # select one op:
                    add_op = random.choice(sequential_op)
                    new_ops.insert(y, add_op)
                    idle_VERTICES -= 1
                    # remove original connection
                    new_matrix[x, y] = 0
                    # expand the matrix size
                    new_matrix = np.insert(np.insert(new_matrix, y, 0, axis=1), y, 0, axis=0)
                    # add new connections between the new op and original ops
                    # len(new_matrix)-2 is the index of inserted column
                    add_conn_idx1 = tuple([x, y])
                    new_matrix[add_conn_idx1] = 1

                    add_conn_idx2 = tuple([y, y + 1])

                    new_matrix[add_conn_idx2] = 1
                    # 缺少并行连接层的添加的代码
        # Layer branch adding 层分支添加
        if new_matrix.sum() < MAX_EDGES and idle_VERTICES > 0: # 边数未达到最大值(9)，且空闲结点数大于0
            # random select 2 ops and add parallel op

            [x, y] = sorted(random.sample(para_list, 2))
            op_add_prob = mutation_rate / 5  # 2 options: parallel adding and no change 5是怎么来的
            # add one parallel op:
            # The difference b
            if random.random() < op_add_prob:
                # select one op:
                add_op = random.choice(parallel_op) # 从三种并行操作([CONV3X3, CONV1X1, MAXPOOL3X3])中随机选择一种
                new_ops.insert(y, add_op)
                idle_VERTICES -= 1
                # expand the matrix size 在 new_matrix y行、y列插入全零向量
                new_matrix = np.insert(np.insert(new_matrix, y, 0, axis=1), y, 0, axis=0)

                add_conn_idx1 = tuple([x, y])
                new_matrix[add_conn_idx1] = 1

                add_conn_idx2 = tuple([y, y + 1])

                new_matrix[add_conn_idx2] = 1

        # Shortcut adding 捷径添加
        # In expectation, V edges flipped (note that most end up being pruned).
        if new_matrix.sum() < MAX_EDGES:
            NUM_VERTICES = new_matrix.shape[0]
            edge_mutation_prob = mutation_rate / 4 # 4是怎么来的
            for src in range(0, NUM_VERTICES - 2):
                for dst in range(src + 1, NUM_VERTICES - 1):
                    if random.random() < edge_mutation_prob and new_matrix[src, dst] == 0: # 原本两层之间没有连接关系
                        new_matrix[src, dst] = 1
                        # residual connection
            if random.random() < edge_mutation_prob and new_matrix[0, -1] == 0: # 输入层和输出层也有可能建立捷径
                new_matrix[0, -1] = 1

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            if len(new_spec.original_ops) != len(old_spec.original_ops):
                return new_spec # 如果新的结构的层数和原来的不一样，一定是新的结构
            elif np.any(
                    new_spec.original_matrix != old_spec.original_matrix) or new_spec.original_ops != old_spec.original_ops:
                return new_spec # 如果新的结构的层数和原来的一样，但是结构不一样，一定是新的结构
            elif new_matrix.sum() == MAX_EDGES or CONV1X1 not in new_spec.original_ops:
                print("Avoid getting stuck in an infinite loop") # 如果新的结构的层数、结构和原来的一样，且边数已经达到了最大值，也没有CONV1X1
                old_spec = victim_spec # 此时结构已经没有编译空间了，应当将最初的结构作为变异的起点


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)