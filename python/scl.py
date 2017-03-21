import os, sys, ctypes
import numpy as np

class params(ctypes.Structure):
    _fields_  = [('X_n', ctypes.c_int),
            ('X_m', ctypes.c_int),
            ('dict_size', ctypes.c_int),
            ('target_sparsity', ctypes.c_int),
           ('max_iterations', ctypes.c_int),
           ('sparse_updater', ctypes.c_char_p),
           ('dict_updater', ctypes.c_char_p),
           ('metric', ctypes.c_char_p),
           ('print_time', ctypes.c_bool),
                 ('normalize_X', ctypes.c_bool),
            ]

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
dll_path = [os.path.join(sys.prefix, 'scl'), os.path.join(curr_path, '../lib/')]

if os.name == 'nt':
    dll_path = [os.path.join(p, 'scl.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'scl.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

if len(lib_path) is 0:
    print('Could not find shared library path at the following locations:')
    print(dll_path)

_mod = ctypes.cdll.LoadLibrary(lib_path[0])
_sparse_code = _mod.sparse_code
_sparse_code.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_int), params]

def as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def sparse_code(X, dict_size, target_sparsity, max_iterations, sparse_updater='mp', dict_updater='gd', metric='rmse', print_time=False):
    X = np.asfortranarray(X, dtype=np.float32)
    D = np.empty((X.shape[0], dict_size), dtype=np.float32, order='F')
    S = np.empty((dict_size, X.shape[1]), dtype=np.float32, order='F')
    param = params()
    param.X_m = X.shape[0]
    param.X_n = X.shape[1]
    param.max_iterations = max_iterations
    param.target_sparsity = target_sparsity
    param.dict_size = dict_size
    param.sparse_updater = sparse_updater.encode('utf-8')
    param.dict_updater = dict_updater.encode('utf-8')
    param.metric = metric.encode('utf-8')
    param.print_time = print_time

    metrics = np.zeros(max_iterations, dtype=np.float32)
    n_metrics = ctypes.c_int()
    _sparse_code(as_fptr(X), as_fptr(D), as_fptr(S), as_fptr(metrics), ctypes.byref(n_metrics), param)

    return D, S, metrics

#Generate a random dictionary and sparse coding and use it to generate X
#X has zero mean and unit variance
def generate_X(m, n, dict_size, target_sparsity, seed=0):
    rs = np.random.RandomState(seed)
    D = rs.normal(size = (m, dict_size) )
    S = np.zeros( (dict_size, n) )
    S[:target_sparsity,:] = rs.normal(size = (target_sparsity, n) )
    for i in range(0, n):
        rs.shuffle(S[:,i])
    X = np.dot(D, S)
    return X/X.std()

def rmse(X, D, S):
    R = X - D.dot(S)
    return np.sqrt((R ** 2).mean())
