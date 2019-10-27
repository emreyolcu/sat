from __future__ import print_function

import argparse
import os
import pdb
import subprocess
import networkx as nx


def create_sat_problem(filename, problem, graph_generator, k):
    while True:
        nx.write_gml(graph_generator(), 'tmp.gml')
        subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf', problem, '--input', 'tmp.gml', str(k)] +
                        (['0'] if problem == 'kcolor' else []))
        try:
            subprocess.check_call(['minisat', 'tmp.cnf'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:
                os.rename('tmp.cnf', filename)
                os.remove('tmp.gml')
                return
            os.remove('tmp.cnf')
            os.remove('tmp.gml')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='output directory')
    parser.add_argument('problem', type=str)
    parser.add_argument('k', type=int)
    parser.add_argument('graph', type=str)
    parser.add_argument('--x', type=float)
    parser.add_argument('--y', type=float)
    parser.add_argument('--z', type=float)
    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError:
        if not os.path.isdir(args.dir):
            raise

    os.chdir(args.dir)

    if args.graph == 'ba':
        gen = lambda: nx.barabasi_albert_graph(int(args.x), int(args.y))
    elif args.graph == 'rr':
        gen = lambda: nx.random_regular_graph(int(args.x), int(args.y))
    elif args.graph == 'rg':
        gen = lambda: nx.random_geometric_graph(int(args.x), args.y)
    elif args.graph == 'ws':
        gen = lambda: nx.watts_strogatz_graph(int(args.x), int(args.y), args.z)

    for i in range(5):
        filename = 'id={}_{}_{}.cnf'.format(i, args.graph, args.k)
        create_sat_problem(filename, args.problem, gen, args.k)


if __name__ == '__main__':
    main()
