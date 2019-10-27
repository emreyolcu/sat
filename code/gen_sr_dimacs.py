import math
import os
import numpy as np
import random
import argparse
import PyMiniSolvers.minisolvers as minisolvers

def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

def mk_out_filenames(opts, n_vars, t):
    prefix = "{}/id={}_n={}".format(opts.out_dir, t, n_vars)
    return ("", "{}.cnf".format(prefix))

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

def gen_iclause_pair(opts):
    n = random.randint(opts.min_n, opts.max_n)

    solver = minisolvers.MinisatSolver()
    for i in range(n): solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < opts.p_k_2 else 2
        k = k_base + np.random.geometric(opts.p_geo)
        iclause = generate_k_iclause(n, k)

        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', action='store', type=str)
    parser.add_argument('n_pairs', action='store', type=int)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)

    parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)

    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)
    opts = parser.parse_args()

    os.makedirs(opts.out_dir)

    if opts.py_seed is not None: random.seed(opts.py_seed)
    if opts.np_seed is not None: np.random.seed(opts.np_seed)

    for pair in range(opts.n_pairs):
        if pair % opts.print_interval == 0: print("[%d]" % pair)
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(opts)
        out_filenames = mk_out_filenames(opts, n_vars, pair)

        iclauses.append(iclause_unsat)
        # write_dimacs_to(n_vars, iclauses, out_filenames[0])

        iclauses[-1] = iclause_sat
        write_dimacs_to(n_vars, iclauses, out_filenames[1])
