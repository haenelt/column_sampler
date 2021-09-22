# -*- coding: utf-8 -*-
"""Bla."""

__all__ = ["read_anchor"]


def read_anchor(file_in):
    with open(file_in) as fp:
        rows = fp.readlines()
        res = {}
        for i, r in enumerate(rows):
            tmp = r.strip()
            res[str(i)] = [int(e) if e.isdigit() else e for e in tmp.split(',')]

    return res


def load_coords():
    pass


def load_data():
    pass


def save_coords():
    pass


def save_data():
    pass


if __name__ == "__main__":
    f = "/home/daniel/Schreibtisch/bla.txt"
    A = read_anchor(f)
