# input list of arrays
# for each array nan values below threshold
# for all arrays nan values which have inconsistent signs

import numpy as np


def make_template(arr, threshold=1.7):
    nrows, ncols = np.shape(arr)
    res = np.ones(nrows)

    if threshold:
        for i in range(ncols):
            res[arr[:, i] < threshold] = 0

    for i in range(ncols):
        tmp = np.sign(arr[:, 0]) * np.sign(arr[:, i])
        res[tmp != 1] = 0

    return res
