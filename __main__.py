#!/usr/local/bin/python3

import argparse
import konch
from importlib import reload
import numpy as np

from crystal import *
from crystal import md, phonon, tools, analyze


def main():
    parser = init_parser()
    args = parser.parse_args()
    if not args.poscar and not args.xdatcars:
        parser.print_usage()
        return
    if args.printlib:
        from crystal import plot
        import matplotlib.pyplot as plt
    if args.poscar:
        print('>>> crys = Crystal("{}")'.format(args.poscar))
        crys = Crystal(args.poscar)
    if args.xdatcars:
        print('>>> dyn = md.DynCrystal({}, outcars={})'.format(args.xdatcars, args.outcars))
        dyn = md.DynCrystal(args.xdatcars, outcars=args.outcars)

    # if a startup file has been passed, execute it
    if args.file:
        exec(open(args.file).read())

    # set up the interactive python interpreter
    print('\n')
    vars = globals()
    vars.update(locals())
    del vars["vars"]
    konch.config({"context": vars,
                  "banner": "Python crystal module",
                  "context_format": 'hide',
                  "ipy_colors": "linux",
                  "ipy_highlighting_style": "monokai"})
    konch.start()


def init_parser():
    parser = argparse.ArgumentParser(description='Python module for POSCAR and XDATCAR manipulation')
    parser.add_argument('poscar', nargs='?', type=str, metavar='POSCAR', help='POSCAR that should be read')
    parser.add_argument('--md', nargs='*', type=str, metavar='XDATCAR', dest='xdatcars', help='Number of XDATCARs that should be read')
    parser.add_argument('--outcar', nargs='*', type=str, metavar='OUTCAR', dest='outcars', help='Number of OUTCARs that should be read together with XDATCARs')
    parser.add_argument('-p', '--print', action='store_true', dest='printlib', help='load libraries <crystal.print> and <matplotlib.pyplot> as plt')
    parser.add_argument('-f', '--file', type=str, metavar='FILE', dest='file', help='execute FILE before starting shell')
    return parser


if __name__ == '__main__':
    main()
