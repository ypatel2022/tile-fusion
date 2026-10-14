
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scipy.sparse.linalg as splinalg
import time
import sys


def generate_banded_matrix(n):
    triplet_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                triplet_list.append((i, j, 1))
            elif i == j + 1 or i == j + 2 or i == j + 3 or i == j - 1 or i == j - 2 or i == j - 3:
                triplet_list.append((i, j, 1))
    return sp.coo_matrix((np.ones(len(triplet_list)), (np.array(triplet_list)[:, 0], np.array(triplet_list)[:, 1])),
                            shape=(n, n))


def generate_banded_matrix_efficient(n, bandwidth):
    triplet_list = []
    half = (bandwidth-1)/2
    for i in range(n):
        min_idx = int(max(0, i-half))
        max_idx = int(min(n, i+half+1))
        for j in range(min_idx, max_idx):
            triplet_list.append((i, j, 1))
    return sp.coo_matrix((np.ones(len(triplet_list)), (np.array(triplet_list)[:, 0], np.array(triplet_list)[:, 1])),
                            shape=(n, n))


def generate_block_diagonal_matrix(n, block_size):
    rows, cols = [], []
    for start in range(0, n, block_size):
        block = range(start, min(start + block_size, n))
        for i in block:
            for j in block:
                rows.append(i)
                cols.append(j)
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))



# main function
if __name__ == '__main__':
    mat = generate_banded_matrix_efficient(int(sys.argv[1]), int(sys.argv[2]))
    # save mat as mtx file
    sio.mmwrite(sys.argv[3], mat)

