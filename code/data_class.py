import os
import pdb
import random
from collections import namedtuple

import numpy as np
import scipy.sparse as sparse
import torch

from cnf import CNF
from util import DataSample, adj, adj_batch, init_edge_attr, to_sparse_tensor

Batch = namedtuple('Batch', ['x', 'adj', 'y', 'idx', 'sizes'])


def load_dir(path, no_sat=False):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        sat = int(not name.startswith('uu')) if not no_sat else -1
        data.append(DataSample(filename, f, adj(f), sat))
    return data


def init_tensors(adj, device):
    n, m = adj[0].shape[0], adj[0].shape[1]
    xv = torch.tensor([1, 0], dtype=torch.float32).repeat(n, 1).to(device)
    xc = torch.tensor([0, 1], dtype=torch.float32).repeat(m, 1).to(device)
    xev = init_edge_attr(n).to(device)
    xec = init_edge_attr(m).to(device)
    vadj = to_sparse_tensor(sparse.hstack(adj)).to(device)
    cadj = to_sparse_tensor(sparse.vstack(adj)).t().to(device)
    return (xv, xc, xev, xec), (vadj, cadj)


def init_batch(batch, device):
    xvs = []
    xcs = []
    vadjs = []
    cadjs = []
    ys = []
    vsizes = []
    csizes = []

    for sample in batch:
        n, m = sample.adj[0].shape[0], sample.adj[0].shape[1]
        xvs.append(torch.tensor([1, 0], dtype=torch.float32).repeat(n, 1))
        xcs.append(torch.tensor([0, 1], dtype=torch.float32).repeat(m, 1))
        vadjs.append(sample.adj)
        cadjs.append(sample.adj)
        ys.append(sample.sat)
        vsizes.append(n)
        csizes.append(m)

    xv = torch.cat(xvs, dim=0).to(device)
    xc = torch.cat(xcs, dim=0).to(device)
    xev = init_edge_attr(np.sum(vsizes)).to(device)
    xec = init_edge_attr(np.sum(csizes)).to(device)
    vadj = to_sparse_tensor(adj_batch(vadjs, sparse.hstack)).to(device)
    cadj = to_sparse_tensor(adj_batch(cadjs, sparse.vstack)).t().to(device)
    y = torch.tensor(ys, dtype=torch.float32).to(device)
    vidx = torch.tensor(np.cumsum([0] + vsizes), dtype=torch.int64).to(device)
    cidx = torch.tensor(np.cumsum([0] + csizes), dtype=torch.int64).to(device)
    vsizes = torch.tensor(vsizes, dtype=torch.float32).to(device)
    csizes = torch.tensor(csizes, dtype=torch.float32).to(device)

    return Batch((xv, xc, xev, xec), (vadj, cadj), y, (vidx, cidx), (vsizes, csizes))


def create_batches(data, batch_size, device):
    data_length = len(data)
    data_indices = list(range(data_length))
    random.shuffle(data_indices)
    for i in range(0, data_length, batch_size):
        yield init_batch([data[j] for j in data_indices[i : i + batch_size]], device)
