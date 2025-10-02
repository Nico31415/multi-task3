import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

c_init = 10**-5
scaling_init = 1e-3
lmdas_init = [-1e-5, 0]  # [0, -1e-5] for different lambda values
# lmdas_init = [0]


argparse_array = ArgparseArray(
    seed=list(range(6)),
    inp_dim=[1000],
    active_dim=[40],
    n_train=[1024],
    # n_train=[2**i for i in range(5, 11)],
    scaling=[1e-3],
    threshold=1e-10,
    epochs=int(1e6),
    lr = .5,
    init_method = ['complex', 'simple'],
    lmda = [f"{val:.10f}" for val in lmdas_init],
    c = [f"{c_init:.10f}"],
    save_folder=name_instance('scaling', 'seed', 'n_train', 'active_dim', 'init_method', 'lmda', 'c', base_folder='data/diagonal/pretrain')
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_pretrain.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
