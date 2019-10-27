from __future__ import print_function

import argparse
import os
import pdb
import subprocess


def create_sat_problem(filename, n, p, k):
    while True:
        subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf', 'domset', '--gnp', str(n), str(p), str(k)])
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
    parser.add_argument('n', type=int, help='number of nodes')
    parser.add_argument('p', type=float, help='probability of edge')
    parser.add_argument('k', type=int, help='size of the dominating set')
    parser.add_argument('id', type=int, help='starting id')
    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError:
        if not os.path.isdir(args.dir):
            raise

    os.chdir(args.dir)

    for i in range(args.N):
        filename = 'id={}_n={}_p={}_k={}.cnf'.format(args.id + i, args.n, args.p, args.k)
        create_sat_problem(filename, args.n, args.p, args.k)


if __name__ == '__main__':
    main()
