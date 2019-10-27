import argparse
import logging
import os
import pdb
import random

import numpy as np

from cnf import CNF

logger = logging.getLogger(__name__)


class WalkSAT:
    def __init__(self, max_tries, max_flips, p):
        self.max_tries = max_tries
        self.max_flips = max_flips
        self.p = p
        self.flips_to_solution = []
        self.backflips = []
        self.unsat_clauses = []

    def compute_true_lit_count(self, clauses):
        n_clauses = len(clauses)
        true_lit_count = [0] * n_clauses
        # true_lit_count = np.zeros(n_clauses)
        for index in range(n_clauses):
            for literal in clauses[index]:
                if self.model[abs(literal)] == literal:
                    true_lit_count[index] += 1
        return true_lit_count

    def select_literal(self, unsat_clause, occur_list):
        broken_min = float('inf')
        min_breaking_lits = []
        nonrandom_selection = False
        if random.random() < self.p:
            min_breaking_lits = unsat_clause
        else:
            nonrandom_selection = True
            for literal in unsat_clause:
                broken_count = 0
                for index in occur_list[-literal]:
                    if self.true_lit_count[index] == 1:
                        broken_count += 1
                    if broken_count > broken_min:
                        break
                if broken_count < broken_min:
                    broken_min = broken_count
                    min_breaking_lits = [literal]
                elif broken_count == broken_min:
                    min_breaking_lits.append(literal)
        return random.choice(min_breaking_lits), nonrandom_selection

    def flip(self, literal, occur_list):
        for index in occur_list[literal]:
            self.true_lit_count[index] += 1
        for index in occur_list[-literal]:
            self.true_lit_count[index] -= 1
        self.model[abs(literal)] *= -1

    def run(self, formula):
        n_clauses = len(formula.clauses)
        for i in range(1, self.max_tries + 1):
            #print('\nnew try')
            self.model = [
                x if random.random() < 0.5 else -x for x in range(formula.n_variables + 1)
            ]
            self.true_lit_count = self.compute_true_lit_count(formula.clauses)
            j = 0
            flipped = set()
            backflipped = 0
            unsat_clauses = []
            while j < self.max_flips:
                # print(self.model)
                unsat_clause_indices = [k for k in range(n_clauses) if self.true_lit_count[k] == 0]
                #print('unsat: ' + str([formula.clauses[k] for k in unsat_clause_indices]))
                unsat_clauses.append(len(unsat_clause_indices))
                # print(len(unsat_clause_indices))
                # unsat_clause_indices = np.where(self.true_lit_count == 0)[0]
                if not unsat_clause_indices:
                # if unsat_clause_indices.size == 0:
                    # logging.info(f'Try: {i}, Flip: {j}')
                    # logging.info(f'Found solution: {self.model[1:]}')
                    break
                selected_clause = random.choice(unsat_clause_indices)
                #print('clause: ' + str(formula.clauses[selected_clause]))
                selected_literal, nonrandom = self.select_literal(
                    formula.clauses[selected_clause], formula.occur_list
                )
                if nonrandom:
                    var = abs(selected_literal)
                    if var not in flipped:
                        flipped.add(var)
                    else:
                        backflipped += 1
                else:
                    pass
                    #print('random')
                self.flip(selected_literal, formula.occur_list)
                j += 1
            self.flips_to_solution.append(j)
            self.backflips.append(backflipped)
            self.unsat_clauses.append(unsat_clauses)


def main(args):
    if args.seed > -1:
        random.seed(args.seed)

    if args.dir_path:
        med_flips = []
        avg_flips = []
        max_flips = []
        solved = []
        med_backflips = []
        avg_backflips = []
        max_backflips = []
        unsats = []
        mean_delta = []
        downwards = []
        upwards = []
        sideways = []
        for i, filename in enumerate(os.listdir(args.dir_path)):
            if i >= args.samples:
                break
            formula = CNF.from_file(os.path.join(args.dir_path, filename))
            walksat = WalkSAT(args.max_tries, args.max_flips, args.p)
            walksat.run(formula)
            flips = walksat.flips_to_solution
            backflips = walksat.backflips
            unsats = walksat.unsat_clauses
            diffs = [np.diff(u) for u in unsats]
            diff_a = np.array(diffs)
            upwards.append(np.mean([np.sum(a > 0) for a in diff_a]))
            downwards.append(np.mean([np.sum(a < 0) for a in diff_a]))
            sideways.append(np.mean([np.sum(a == 0) for a in diff_a]))
            mean_delta.append(np.mean([np.mean(d) for d in diffs]))
            med_backflips.append(np.median(backflips))
            avg_backflips.append(np.mean(backflips))
            max_backflips.append(np.max(backflips))
            med = np.median(flips)
            med_flips.append(med)
            avg_flips.append(np.mean(flips))
            max_flips.append(np.max(flips))
            solved.append(int(med < args.max_flips))
        # print(med_flips)
        # print(avg_flips)
        # print(max_flips)
        logging.info(
            'Acc: {:.4f}, Med: {:.4f}, Mean: {:.4f}, Max: {:.4f}'.format(
                100 * np.mean(solved), np.median(med_flips), np.mean(avg_flips), np.mean(max_flips)
            )
        )
        logging.info(
            '(Backflips) Med: {:.4f}, Mean: {:.4f}, Max: {:.4f}'.format(
                np.median(med_backflips), np.mean(avg_backflips), np.mean(max_backflips)
            )
        )
        logging.info('(Delta) Mean: {:.4f}'.format(np.mean(mean_delta)))
        logging.info('(Movement) Up: {:.4f}, Side: {:.4f}, Down: {:.4f}'.format(np.mean(upwards),
                                                                                np.mean(sideways), np.mean(downwards)))
    elif args.file_path:
        formula = CNF.from_file(args.file_path)
        walksat = WalkSAT(args.max_tries, args.max_flips, args.p)
        walksat.run(formula)
        logging.info(
            'Number of solutions found: {}\n'
            'Avg, std of flips to solution: {:.4f}, {:.4f}'.format(
                len(walksat.flips_to_solution),
                np.mean(walksat.flips_to_solution),
                np.std(walksat.flips_to_solution),
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    main(args)
