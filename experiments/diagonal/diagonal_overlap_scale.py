import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    active_dim_1=40,
    active_dim_2=[5],#[5, 10, 20, 30, 40],
    scaling=(lambda array_id, w_scaling, **kwargs: {1e-1: 1e-4, 1e-2: 1e-5, 1e-3: 1e-6, 1e-4: 1e-7}[w_scaling]),
    model_scaling=(lambda array_id, w_scaling, **kwargs: {1e-1: 1e-4, 1e-2: 1e-5, 1e-3: 1e-6, 1e-4: 1e-7}[w_scaling]),
    w_scaling=[1e-1, 1e-2, 1e-3, 1e-4],
    inp_dim=1000,
    model_path=(lambda array_id, seed, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--n_train=1024--scaling=0.001/model.pt'),
    threshold=1e-10,
    epochs=int(1e5),
    load_model=[True, False],
    one_task=[True],
    linear_readout=[False],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 10)],
    aux_overlap_bool=['yes', 'no'],
    overlap=(lambda array_id, overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    lr=.1,
    save_path=name_instance('seed', 'n_train2', 'active_dim_2', 'load_model', 'linear_readout', 'one_task', 'overlap_bool', 'w_scaling',
                            base_folder='data/diagonal/overlap_scale'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
