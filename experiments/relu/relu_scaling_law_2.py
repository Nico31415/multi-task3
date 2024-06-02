import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    n_units=[6],
    orthogonal=True,
    n_train=[2**i for i in range(5, 11)],
    scaling=[1e-3, 1.],
    threshold=1e-8,
    epochs=int(1e5),
    inp_dim=15, 
    lr = (lambda array_id, scaling, **kwargs: {1.: 0.001, 1e-3: 0.1}[scaling]),
    save_folder=name_instance('scaling', 'seed', 'n_units', 'n_train', base_folder='data/relu/scaling_law')
)

def main(args):
    argparse_array.call_script('experiments/iclr_experiments/relu_scaling_law.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
