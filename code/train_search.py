import functools
import logging
import pdb
import pickle
import random
from os.path import join
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process

import evaluate
import util
from data_search import load_dir
# from gnn import A2CPolicy, ReinforcePolicy, PGPolicy
from search import LocalSearch
from util import normalize

logger = logging.getLogger(__name__)


train_stats = {'iter': [], 'avg': [], 'med': [], 'acc': [], 'max': []}
eval_stats = {'iter': [], 'avg': [], 'med': [], 'acc': [], 'max': []}


def log(epoch, batch_count, avg_loss, avg_acc):
    logger.info(
        'Epoch: {:4d},  Iter: {:8d},  Loss: {:.4f},  Acc: {:.4f}'.format(
            epoch, batch_count, avg_loss, avg_acc
        )
    )


def load_data(path, train_sets, eval_set, shuffle=False):
    train_len = 0
    for train_set in train_sets:
        train_set['data'] = load_dir(join(path, train_set['name']))[: train_set['samples']]
        if shuffle:
            random.shuffle(train_set['data'])
        train_len += len(train_set['data'])
    if eval_set:
        eval_set['data'] = load_dir(join(path, eval_set['name']))[: eval_set['samples']]
        if shuffle:
            random.shuffle(eval_set['data'])

    logger.info('Loaded {} training problems from {}'.format(train_len, path))
    if eval_set:
        logger.info('Loaded {} evaluation problems from {}'.format(len(eval_set['data']), path))

    return (train_sets, eval_set)


def reinforce(sat, history, config):
    log_probs_list = history[0]
    T = len(log_probs_list)

    log_probs_filtered = []
    mask = np.zeros(T, dtype=np.uint8)
    for i, x in enumerate(log_probs_list):
        if x is not None:
            log_probs_filtered.append(x)
            mask[i] = 1

    log_probs = torch.stack(log_probs_filtered)
    partial_rewards = config['discount'] ** torch.arange(T - 1, -1, -1, dtype=torch.float32, device=log_probs.device)

    return -torch.mean(partial_rewards[torch.from_numpy(mask).to(log_probs.device)] * log_probs)


def pg(sat, history, config):
    log_probs_list, values_list, entropies_list = history
    T = len(log_probs_list)

    log_probs_filtered = []
    values_filtered = []
    entropies_filtered = []
    mask = np.zeros(T, dtype=np.uint8)
    for i, (x, y, z) in enumerate(zip(log_probs_list, values_list, entropies_list)):
        if x is not None:
            log_probs_filtered.append(x)
            values_filtered.append(y)
            entropies_filtered.append(z)
            mask[i] = 1

    log_probs = torch.stack(log_probs_filtered)
    values = torch.stack(values_filtered)
    entropies = torch.stack(entropies_filtered)

    partial_rewards = config['discount'] ** torch.arange(T - 1, -1, -1, dtype=torch.float32, device=log_probs.device)
    rewards_masked = partial_rewards[torch.from_numpy(mask).to(log_probs.device)]

    actor_loss = -torch.mean((rewards_masked - values.detach()) * log_probs)
    critic_loss = F.mse_loss(values, rewards_masked)
    entropy_loss = -torch.mean(entropies)

    return actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss


def a2c(sat, history, config):
    log_probs, values, entropies = history
    device = log_probs[0].device
    N = config['a2c_n']

    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    entropies = torch.stack(entropies)

    rewards = torch.zeros_like(log_probs)
    rewards[-1] = int(sat)

    T = rewards.shape[0]
    if N < T:
        R = (config['discount'] ** N) * torch.cat([values.detach()[N:], torch.zeros(N).to(device)])
    else:
        R = torch.zeros(T).to(device)

    partial_rewards = torch.zeros_like(rewards)
    for t in range(T):
        M = min(t + N, T)
        ps = torch.arange(M - t, dtype=torch.float32, device=device)
        partial_rewards[t] = torch.sum((config['discount'] ** ps) * rewards[t:M])
    R += partial_rewards

    actor_loss = -torch.mean((R - values.detach()) * log_probs)
    critic_loss = F.mse_loss(values, R)
    entropy_loss = -torch.mean(entropies)

    return actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss


# def wrap_single(dummy, ls, sample, discount):
#     sat, log_probs = ls.generate_episode(sample)
#     loss = reinforce_loss(sat, log_probs, discount) if log_probs else None
#     return loss.detach(), len(log_probs)


def generate_episodes(ls, sample, max_tries, max_flips, config):
    # f = functools.partial(wrap_single, ls=ls, sample=sample, discount=discount)
    # pool = Pool(processes=max_tries)
    # res = pool.map(f, range(max_tries))
    # pdb.set_trace()
    if config['method'] == 'reinforce':
        loss_fn = reinforce
    elif config['method'] == 'reinforce_multi':
        loss_fn = reinforce
    elif config['method'] == 'pg':
        loss_fn = pg
    elif config['method'] == 'a2c':
        loss_fn = a2c

    flips = []
    losses = []

    for _ in range(max_tries):
        sat, flip, history = ls.generate_episode(sample, max_flips, config['walk_prob'])
        flip = flip[0]
        if sat and flip > 0 and not all(map(lambda x: x is None, history[0])):
            losses.append(loss_fn(sat, history, config))
        flips.append(flip)
    return losses, flips


def stats_better(new, old):
    return new[0] <= old[0] and new[1] <= old[1] and new[2] <= old[2] and new[3] >= old[3]


def flip_update(fp, flips, max_flips):
    mf, af, xf, sv = fp
    med = np.median(flips)
    mf.append(med)
    af.append(np.mean(flips))
    xf.append(np.max(flips))
    sv.append(int(med < max_flips))


def flip_report(header, fp):
    mf, af, xf, sv = fp
    m = np.median(mf)
    a = np.mean(af)
    ax = np.mean(xf)
    acc = 100 * np.mean(sv)
    logger.info(
        f'{header}  Acc: {acc:10.2f},  Flips: {m:10.2f} (med) / {a:10.2f} (mean) / {ax:10.2f} (max)'
    )
    return ([], [], [], []), (m, a, ax, acc)


def eval(ls, eval_set, config):
    ls.policy.eval()
    with torch.no_grad():
        fp = ([], [], [], [])
        for sample in eval_set['data']:
            if config['eval_multi']:
                flips = evaluate.generate_episodes(ls, sample, eval_set['max_tries'], eval_set['max_flips'], config['walk_prob'], False)[0]
            else:
                _, flips = generate_episodes(ls, sample, eval_set['max_tries'], eval_set['max_flips'], config)
            flip_update(fp, flips, eval_set['max_flips'])
    _, stats = flip_report(f'(Eval)  ', fp)
    return stats


def train(ls, optimizer, scheduler, data, config):
    train_set, eval_set = data

    fp = ([], [], [], [])
    stats = (float('inf'), float('inf'), float('inf'), 0)

    if not config['no_eval']:
        new_stats = eval(ls, eval_set, config)
        m, a, ax, acc = new_stats
        eval_stats['iter'].append(0)
        eval_stats['avg'].append(a)
        eval_stats['med'].append(m)
        eval_stats['acc'].append(acc)
        eval_stats['max'].append(ax)
        if stats_better(new_stats, stats):
            logger.info('Saving best model parameters')
            torch.save(ls.policy, join(config['dir'], 'model_best.pth'))
            stats = new_stats

    for i in range(1, train_set['iterations'] + 1):
        ls.policy.train()

        losses, flips = generate_episodes(
            ls,
            train_set['data'][i % len(train_set['data'])],
            train_set['max_tries'],
            train_set['max_flips'],
            config,
        )
        flip_update(fp, flips, train_set['max_flips'])

        if losses:
            loss = torch.stack(losses).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if i % config['report_interval'] == 0:
            fp, new_stats = flip_report(f'Iter: {i:6d},', fp)
            m, a, ax, acc = new_stats
            train_stats['iter'].append(i)
            train_stats['avg'].append(a)
            train_stats['med'].append(m)
            train_stats['acc'].append(acc)
            train_stats['max'].append(ax)
            # if stats_better(new_stats, stats):
            #     logger.info('Saving best model parameters')
            #     torch.save(ls.policy, join(config['dir'], 'model_best.pth'))
            #     stats = new_stats

        if i % config['save_interval'] == 0:
            torch.save(ls.policy, join(config['dir'], 'model_last.pth'))
            logger.info('Saving last model parameters')
            pickle.dump(train_stats, open(join(config['dir'], 'train_stats.pkl'), 'wb'))
            pickle.dump(eval_stats, open(join(config['dir'], 'eval_stats.pkl'), 'wb'))
            logger.info('Saving stats')

        if not config['no_eval'] and i % config['eval_interval'] == 0:
            new_stats = eval(ls, eval_set, config)
            m, a, ax, acc = new_stats
            eval_stats['iter'].append(i)
            eval_stats['avg'].append(a)
            eval_stats['med'].append(m)
            eval_stats['acc'].append(acc)
            eval_stats['max'].append(ax)
            if stats_better(new_stats, stats):
                logger.info('Saving best model parameters')
                torch.save(ls.policy, join(config['dir'], 'model_best.pth'))
                stats = new_stats

def main():
    config, device = util.setup()
    logger.setLevel(getattr(logging, config['log_level'].upper()))
    gnn = import_module('gnn' if config['mlp_arch'] else 'gnn_old')

    if config['method'] == 'reinforce':
        model = gnn.ReinforcePolicy
    elif config['method'] == 'reinforce_multi':
        model = gnn.ReinforcePolicy
    elif config['method'] == 'pg':
        model = gnn.PGPolicy
    elif config['method'] == 'a2c':
        model = gnn.A2CPolicy

    if config['model_path']:
        logger.info('Loading model parameters from {}'.format(config['model_path']))
        policy = torch.load(config['model_path']).to(device)

        if config['load_with_noise']:
            with torch.no_grad():
                # for p in policy.parameters():
                #     p.add_(torch.randn(p.size()) * 0.02)
                for p in policy.policy_readout.parameters():
                    p.add_(torch.randn(p.size()) * 0.1)
    else:
        if config['mlp_arch']:
            policy = model(3, config['gnn_hidden_size'], config['readout_hidden_size'],
                           config['mlp_arch'], config['gnn_iter'], config['gnn_async']).to(device)
        else:
            policy = model(3, config['gnn_hidden_size'], config['readout_hidden_size']).to(device)
    optimizer = getattr(optim, config['optimizer'])(policy.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['lr_milestones'], gamma=config['lr_decay']
    )
    ls = LocalSearch(policy, device, config)

    train_sets, eval_set = load_data(config['data_path'], config['train_sets'], config['eval_set'],
                                     config['data_shuffle'])

    for i in range(1, config['cycles'] + 1):
        logger.info(f'Cycle: {i}')
        for train_set in train_sets:
            logger.info('Train set: {}'.format(train_set['name']))
            train(ls, optimizer, scheduler, (train_set, eval_set), config)


if __name__ == '__main__':
    main()
