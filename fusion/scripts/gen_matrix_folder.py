import sys
import scipy.io as sio
from gen_matrix import generate_banded_matrix_efficient, generate_block_diagonal_matrix
import argparse
import os

def create_matrix_folder(mtx_size_list, folder, bandwidth, block_size):
    matrix_names = []
    os.makedirs(folder, exist_ok=True)
    for s in mtx_size_list:
        for bw in bandwidth:
            mat_name = str(bw) + '_banded_' + str(s) + '.mtx'
            matrix_names.append(mat_name)
            mat = generate_banded_matrix_efficient(s, bw)
            # save mat as mtx file
            sio.mmwrite(os.path.join(folder, mat_name), mat)
        for b in block_size:
            mat_name = str(b) + '_block_diagonal_' + str(s) + '.mtx'
            matrix_names.append(mat_name)
            sio.mmwrite(os.path.join(folder, mat_name),
                        generate_block_diagonal_matrix(s, b))
    with open(os.path.join(folder,"mat_list.txt"), 'w') as mat_list:
        matrix_names = [mn + '\n' for mn in matrix_names]
        mat_list.writelines(matrix_names)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default="./data/tri-banded")
    parser.add_argument('-b', '--bandwidth', nargs='*', type=int, default=[3])
    parser.add_argument('-bd', '--block-size', nargs='+', type=int, default=[])
    parser.add_argument('-sl', '--size-list', nargs='+', type=int)

    args = parser.parse_args()
    create_matrix_folder(args.size_list, args.folder, args.bandwidth,
                         args.block_size)
