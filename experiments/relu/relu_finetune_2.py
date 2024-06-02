import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    n_units_1=6,
    n_units_2=[1,2,3,4,5,6],
    scaling=1e-3,
    model_scaling=1e-3,
    model_path=(lambda array_id, seed, **kwargs: f'data/relu/scaling_law/seed={seed}--n_units=6--n_train=1024--scaling=0.001/model.pt'),
    threshold=1e-8,
    epochs=int(1e5),
    load_model=[False],
    one_task=[False],
    setup=['backprop'],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 10)],
    aux_corr=[0., 1.],
    correlation=(lambda array_id, n_units_2, corr, **kwargs: [corr]*n_units_2),
    train_var=[[1.]*15],
    lr=1.,
    save_path=name_instance('seed', 'n_train2', 'n_units_2', 'corr', 'load_model', 'setup', 'one_task',
                            base_folder='data/relu/relu_finetune_2')
)

def main(args):
    argparse_array.call_script('experiments/relu/relu_finetuning.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
