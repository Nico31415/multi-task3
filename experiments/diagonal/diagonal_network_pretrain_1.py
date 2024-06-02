import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    inp_dim=[1000],
    active_dim=[40,1000],
    n_train=[2**i for i in range(5, 11)],
    scaling=[1e-3, 1.],
    threshold=1e-10,
    epochs=int(1e6),
    lr = .5,
    save_folder=name_instance('scaling', 'seed', 'n_train', 'active_dim', base_folder='data/diagonal/pretrain')
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_pretrain.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
