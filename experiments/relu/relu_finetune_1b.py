import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    n_units_1=6,
    n_units_2=6,
    scaling=1e-3,
    model_scaling=1e-3,
    model_path=(lambda array_id, seed, **kwargs: f'data/relu/scaling_law/seed={seed}--n_units=6--n_train=1024--scaling=0.001/model.pt'),
    threshold=1e-8,
    epochs=int(1e5),
    load_model=[True, False],
    one_task=[True],
    setup=['backprop', 'linear_readout', 'ntk'],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 10)],
    correlation=[[0.]*6, [1.]*5+[0.], [1.]*6, [1.]*3+[0.]*3, [0.9]*6, [1.]*2+[0.9]*2+[0.]*2],
    train_var=[[1.]*15],
    lr=1.,
    save_path=name_instance('seed', 'n_train2', 'correlation', 'load_model', 'setup', 'one_task',
                            base_folder='data/relu/finetune_1')
)

def main(args):
    argparse_array.call_script('experiments/relu/relu_finetuning.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
