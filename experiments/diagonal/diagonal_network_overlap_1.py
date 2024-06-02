import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    active_dim_1=40,
    active_dim_2=40,
    scaling=1e-3,
    model_scaling=1e-3,
    inp_dim=1000,
    model_path=(lambda array_id, seed, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--n_train=1024--scaling=0.001/model.pt'),
    threshold=1e-10,
    epochs=int(1e6),
    load_model=[True, False],
    one_task=[True],
    linear_readout=[True, False],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 10)],
    overlap=[0, 20, 30, 40],
    lr=.1,
    save_path=name_instance('seed', 'n_train2', 'overlap', 'load_model', 'linear_readout', 'one_task',
                            base_folder='data/diagonal/overlap'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
