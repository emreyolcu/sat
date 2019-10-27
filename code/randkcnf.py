from __future__ import print_function

import argparse
import os
import pdb
import subprocess


def create_sat_problem(filename, k, n, m):
    while True:
        subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf', 'randkcnf', str(k), str(n), str(m)])
        try:
            subprocess.check_call(['minisat', 'tmp.cnf'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:
                os.rename('tmp.cnf', filename)
                return
            os.remove('tmp.cnf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='output directory')
    parser.add_argument('N', type=int, help='number of problems to be generated')
    parser.add_argument('k', type=int, help='number of literals in a clause')
    parser.add_argument('n', type=int, help='number of variables')
    parser.add_argument('m', type=int, help='number of clauses')
    parser.add_argument('id', type=int, help='starting id')
    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError:
        if not os.path.isdir(args.dir):
            raise

    os.chdir(args.dir)

    for i in range(args.N):
        filename = 'id={}_n={}_m={}.cnf'.format(args.id + i, args.n, args.m)
        create_sat_problem(filename, args.k, args.n, args.m)


if __name__ == '__main__':
    main()
