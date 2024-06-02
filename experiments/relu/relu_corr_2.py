import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    seed=list(range(6)),
    n_units=6,
    scaling=1e-6,
    model_scaling=1e-6,
    model_path=(lambda array_id, seed, **kwargs: f'data/relu/scaling_law/seed={seed}--scaling=1e-06/model.pt'),
    threshold=1e-10,
    epochs=int(1e5),
    load_model=[False],
    one_task=[False],
    setup=['backprop'],
    n_train1=1024,
    n_train2=[2**i for i in range(5, 11)],
    aux_corr = [0.8, 0.9, 0.95, 0.99, 1.0],
    correlation=(lambda array_id, corr, **kwargs: [corr]*6),
    aux_mag = [1., 0.5, 0.1, 0.05, 0.01],
    magnitude=(lambda array_id, mag, **kwargs: [mag]*6),
    train_var=[[1.]*15],
    lr=1e6,
    save_path=name_instance('seed', 'n_train2', 'corr', 'load_model', 'setup', 'one_task', 'mag',
                            base_folder='data/relu/corr')
)

def main(args):
    argparse_array.call_script('experiments/relu/relu_finetuning_v2.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
