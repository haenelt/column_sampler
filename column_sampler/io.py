
def read_anchor(file_in):
    with open(file_in) as fp:
        rows = fp.readlines()
        res = {}
        for i, r in enumerate(rows):
            tmp = r.strip()
            res[str(i)] = [int(e) if e.isdigit() else e for e in tmp.split(',')]

    return res


f = "/home/daniel/Schreibtisch/bla.txt"
A = read_anchor(f)
