import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

c_init = 10**-5
lmdas_init = [0, -10**-5]

argparse_array = ArgparseArray(
    seed=list(range(6)),
    active_dim_1=40,
    # active_dim_2=[5, 10, 20, 30, 40],
    active_dim_2 = [5, 40],
    scaling=[1e-3],
    w_scaling=[0.01, 1.0],
    init_method=['complex'],
    lmda=[f"{val:.10f}" for val in lmdas_init],
    c=[f"{c_init:.10f}"],
    model_scaling=1e-3,
    inp_dim=1000,
    model_path=(lambda array_id, seed, scaling, init_method, lmda, c, **kwargs: 
                f'data/diagonal/pretrain/seed={seed}--active_dim=40--n_train=1024--scaling={scaling}--init_method={init_method}--lmda={lmda}--c={c}/model.pt'),
    threshold=1e-10,
    epochs=int(1e5),
    # load_model=[True, False],
    load_model=[True],
    one_task=[True],
    linear_readout=[False],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 9)],
    # n_train2 = [16],
    # n_train2 = [16],
    # n_train2=[32, 64],
    aux_overlap_bool=['no', 'yes'],
    overlap=(lambda array_id, overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    lr=.1,
    save_path=name_instance('seed', 'n_train2', 'active_dim_2', 'load_model', 'linear_readout', 'one_task', 'overlap_bool', 'w_scaling', 'scaling', 'init_method', 'lmda', 'c',
                            base_folder='data/diagonal/sparse_overlap'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
