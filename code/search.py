import pdb
import random

import scipy.sparse as sparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from data_search import Batch, DataSample, init_edge_attr, init_tensors, to_sparse_tensor
from util import normalize


class LocalSearch:
    def __init__(self, policy, device, config):
        self.policy = policy
        self.device = device
        if config['method'] == 'reinforce':
            self.generate_episode = self._generate_episode_reinforce
        elif config['method'] == 'reinforce_multi':
            self.generate_episode = self._generate_episode_reinforce_multi
        elif config['method'] == 'pg':
            self.generate_episode = self._generate_episode_pg
        elif config['method'] == 'a2c':
            self.generate_episode = self._generate_episode_a2c

    def eval(self):
        self.generate_episode = self._eval_generate_episode_reinforce

    def train(self):
        self.generate_episode = self._generate_episode_reinforce

    def _eval_select_variable_reinforce(self, data):
        logit = self.policy(data)
        v = logit.argmax(dim=0)
        return v

    # @profile
    def _select_variable_reinforce(self, data):
        logit = self.policy(data)
        prob = F.softmax(logit, dim=0)
        dist = Categorical(prob.view(-1))
        v = dist.sample()
        return v, dist.log_prob(v)

    def _select_variable_reinforce_multi(self, data):
        logit = self.policy(data)
        prob = torch.sigmoid(logit)
        dist = Bernoulli(prob.view(-1))
        vs = dist.sample()
        return vs.nonzero(), dist.log_prob(vs).sum()

    def _select_variable_pg(self, data):
        logit, value = self.policy(data)

        prob = F.softmax(logit, dim=0)
        log_prob = F.log_softmax(logit, dim=0)
        entropy = -(log_prob * prob).sum()

        dist = Categorical(prob.view(-1))
        v = dist.sample()

        return v, dist.log_prob(v), value, entropy

    def _select_variable_a2c(self, data):
        logit, value = self.policy(data)

        prob = F.softmax(logit, dim=0)
        log_prob = F.log_softmax(logit, dim=0)
        entropy = -(log_prob * prob).sum()

        dist = Categorical(prob.view(-1))
        v = dist.sample()

        return v, dist.log_prob(v), value, entropy

    def _eval_generate_episode_reinforce(self, sample, max_flips, walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)
        flip = 0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            if random.random() < walk_prob:
                unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
                v = abs(random.choice(unsat_clause)) - 1
            else:
                v = self._eval_select_variable_reinforce(data)
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1
        return sat, flip, None

    # @profile
    def _generate_episode_reinforce(self, sample, max_flips, walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)
        log_probs = []
        flip = 0
        flipped = set()
        backflipped = 0
        unsat_clauses = []
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            unsat_clauses.append(len(unsat_clause_indices))
            # print(len(unsat_clause_indices))
            sat = not unsat_clause_indices
            if sat:
                break
            if random.random() < walk_prob:
                unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
                v, log_prob = abs(random.choice(unsat_clause)) - 1, None
            else:
                v, log_prob = self._select_variable_reinforce(data)
                if v.item() not in flipped:
                    flipped.add(v.item())
                else:
                    backflipped += 1
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1
            log_probs.append(log_prob)
        # print(backflipped)
        return sat, (flip, backflipped, unsat_clauses), (log_probs,)

    def _generate_episode_reinforce_multi(self, sample, max_flips, walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)
        log_probs = []
        flip = 0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            if random.random() < walk_prob:
                unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
                v, log_prob = abs(random.choice(unsat_clause)) - 1, None
                flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            else:
                vs, log_prob = self._select_variable_reinforce_multi(data)
                flip_multi_(data.x[0], data.sol, true_lit_count, vs, f.occur_list)
            flip += 1
            log_probs.append(log_prob)
        return sat, flip, (log_probs,)

    def _generate_episode_pg(self, sample, max_flips, walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)
        log_probs = []
        values = []
        entropies = []
        flip = 0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            if random.random() < walk_prob:
                unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
                v, log_prob, value, entropy = abs(random.choice(unsat_clause)) - 1, None, None, None
            else:
                v, log_prob, value, entropy = self._select_variable_pg(data)
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
        return sat, flip, (log_probs, values, entropies)

    def _generate_episode_a2c(self, sample, max_flips):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)

        log_probs = []
        values = []
        entropies = []

        flip = 0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            v, log_prob, value, entropy = self._select_variable_a2c(data)
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1

            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

        return sat, flip, (log_probs, values, entropies)


def flip_multi_(xv, sol, true_lit_count, vs, occur_list):
    for v in vs.int().cpu().numpy().flatten():
        lit_false = (v + 1) * normalize(int(xv[v, 0].item() == 1))
        for i in occur_list[lit_false]:
            true_lit_count[i] += 1
        for i in occur_list[-lit_false]:
            true_lit_count[i] -= 1
        xv[v, :2] = 1 - xv[v, :2]
        sol[v + 1] *= -1


def flip_(xv, sol, true_lit_count, v, occur_list):
    lit_false = (v + 1) * normalize(int(xv[v, 0].item() == 1))
    for i in occur_list[lit_false]:
        true_lit_count[i] += 1
    for i in occur_list[-lit_false]:
        true_lit_count[i] -= 1
    xv[v, :2] = 1 - xv[v, :2]
    sol[v + 1] *= -1


def compute_true_lit_count(clauses, sol):
    n_clauses = len(clauses)
    true_lit_count = [0] * n_clauses
    for i in range(n_clauses):
        for lit in clauses[i]:
            if sol[abs(lit)] == lit:
                true_lit_count[i] += 1
    return true_lit_count
