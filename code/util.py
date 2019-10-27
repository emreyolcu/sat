import argparse
import logging
import os
import pdb
import random
from collections import namedtuple
from os.path import join

import numpy as np
import scipy.sparse as sparse
import torch
import yaml

logger = logging.getLogger(__name__)

DataSample = namedtuple('DataSample', ['filename', 'formula', 'adj', 'sat'])


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config_data = f.read()
    config = yaml.load(config_data)

    config['dir'] = join('results', config['name'])
    os.makedirs(config['dir'], exist_ok=True)

    log_file = join(config['dir'], 'train.log')
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    logger.info('Configuration:\n' + config_data)

    if config['seed']:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    use_gpu = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_gpu else 'cpu')
    logger.info(f'Device: {device}')

    config['no_eval'] = not config['eval_set']

    return config, device


def adj_sign(n, m, occur):
    i = np.repeat(range(n), [len(lst) for lst in occur])
    j = np.concatenate(occur)
    v = np.ones(len(i), dtype=np.int64)
    return sparse.coo_matrix((v, (i, j)), shape=(n, m))


def adj(f):
    n, m, occur = f.n_variables, len(f.clauses), f.occur_list
    adj_pos = adj_sign(n, m, occur[1 : n + 1])
    adj_neg = adj_sign(n, m, occur[:n:-1])
    return (adj_pos, adj_neg)


def adj_batch(adjs, fstack):
    adjp, adjn = list(zip(*adjs))
    return fstack((sparse.block_diag(adjp), sparse.block_diag(adjn)))


def to_sparse_tensor(x):
    x = x.tocoo()
    i = torch.tensor(np.vstack((x.row, x.col)), dtype=torch.int64)
    v = torch.tensor(x.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, torch.Size(x.shape))


def init_edge_attr(k):
    return torch.cat(
        (
            torch.tensor([1, 0], dtype=torch.float32).expand(k, 2),
            torch.tensor([0, 1], dtype=torch.float32).expand(k, 2),
        ),
        dim=0,
    )


def normalize(x):
    return 2 * x - 1


def unnormalize(x):
    return (x + 1) / 2
