import functools
import argparse
import logging
import random

import numpy as np
import torch

from data_search import load_dir
from search import LocalSearch
from multiprocessing import Pool

logger = logging.getLogger(__name__)


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


def wrap_single(dummy, ls, sample, max_flips, walk_prob):
    sat, stat, _ = ls.generate_episode(sample, max_flips, walk_prob)
    return stat


def generate_episodes(ls, sample, max_tries, max_flips, walk_prob, no_multi):
    if not no_multi:
        f = functools.partial(wrap_single, ls=ls, sample=sample, max_flips=max_flips, walk_prob=walk_prob)

        with Pool(25) as p:
            stat = p.map(f, range(max_tries))

        return list(zip(*stat))
    else:
        flips = []
        backflips = []
        unsat_clauses = []

        for j in range(max_tries):
            # logger.info(f'Try: {j}')
            sat, stat, _ = ls.generate_episode(sample, max_flips, walk_prob)
            flip, backflip, unsat = stat
            flips.append(flip)
            backflips.append(backflip)
            unsat_clauses.append(unsat)
        return flips, backflips, unsat_clauses


def main(args):
    if args.seed > -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_gpu = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_gpu else 'cpu')
    logger.info(f'Device: {device}')

    policy = torch.load(args.model_path).to(device)
    ls = LocalSearch(policy, device, {'method': 'reinforce'})
    eval_set = load_dir(args.dir_path)

    ls.policy.eval()
    # ls.eval()
    with torch.no_grad():
        fp = ([], [], [], [])
        med_backflips = []
        avg_backflips = []
        max_backflips = []
        unsats = []
        mean_delta = []
        downwards = []
        upwards = []
        sideways = []
        for i, sample in enumerate(eval_set):
            if i % 10 == 0:
                logger.info(f'Sample: {i}')
            if i >= args.samples:
                break
            flips, backflips, unsats = generate_episodes(ls, sample, args.max_tries,
    args.max_flips, args.p, args.no_multi)
            if backflips is not None:
                med_backflips.append(np.median(backflips))
                avg_backflips.append(np.mean(backflips))
                max_backflips.append(np.max(backflips))
            if unsats is not None:
                diffs = [np.diff(u) for u in unsats]
                diff_a = np.array(diffs)
                upwards.append(np.mean([np.sum(a > 0) for a in diff_a]))
                downwards.append(np.mean([np.sum(a < 0) for a in diff_a]))
                sideways.append(np.mean([np.sum(a == 0) for a in diff_a]))
                mean_delta.append(np.mean([np.mean(d) for d in diffs]))
            flip_update(fp, flips, args.max_flips)
            
    _, stats = flip_report(f'(Eval)  ', fp)
    logging.info(
        '(Backflips) Med: {:.4f}, Mean: {:.4f}, Max: {:.4f}'.format(
            np.median(med_backflips), np.mean(avg_backflips), np.mean(max_backflips)
        )
    )
    logging.info('(Delta) Mean: {:.4f}'.format(np.mean(mean_delta)))
    logging.info('(Movement) Up: {:.4f}, Side: {:.4f}, Down: {:.4f}'.format(np.mean(upwards), np.mean(sideways), np.mean(downwards)))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=None)
    parser.add_argument('-f', '--file_path', type=str)
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--no_multi', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    main(args)
