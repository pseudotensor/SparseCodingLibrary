import scl
import numpy as np
from collections import namedtuple
from ksvd import KSVD

Case = namedtuple('Case', ['n', 'm', 'dict_size', 'target_sparsity', 'iterations', 'sparse_updater', 'dict_updater'])

c = Case(6, 5, 6, 3, 10, 'omp2', 'aksvd')
X = scl.generate_X(c.m, c.n, c.dict_size, c.target_sparsity, 13)
scl_result = scl.sparse_code(X, c.dict_size, c.target_sparsity, c.iterations, sparse_updater=c.sparse_updater, dict_updater=c.dict_updater)
print(scl_result[2])
