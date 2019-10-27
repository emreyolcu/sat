import os
import pdb
import random
from collections import namedtuple

import scipy.sparse as sparse
import torch

from cnf import CNF
from util import DataSample, adj, adj_batch, init_edge_attr, to_sparse_tensor

Batch = namedtuple('Batch', ['x', 'adj', 'sol'])


def load_dir(path):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        if name.startswith('uu'):
            continue
        data.append(DataSample(filename, f, adj(f), None))
    return data


def init_tensors(sample, device):
    # [1,0,0] -> assigned False
    # [0,1,0] -> assigned True
    # [0,0,1] -> clause
    adj = sample.adj
    n, m = adj[0].shape[0], adj[0].shape[1]
    xv = torch.zeros(n, 3, dtype=torch.float32)
    sol = [x if random.random() < 0.5 else -x for x in range(n + 1)]
    xv[torch.arange(n), (torch.tensor(sol[1:]) > 0).long()] = 1
    xv = xv.to(device)
    xc = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    xev = init_edge_attr(n).to(device)
    xec = init_edge_attr(m).to(device)
    vadj = to_sparse_tensor(sparse.hstack(adj)).to(device)
    cadj = to_sparse_tensor(sparse.vstack(adj)).t().to(device)
    return Batch((xv, xc, xev, xec), (vadj, cadj), sol)
