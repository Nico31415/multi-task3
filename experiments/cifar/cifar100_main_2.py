import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(50)),
    mode=['finetuning'],
    n_samples=[10, 20, 50, 100, 200, 500, 1000],
    finetune_scaling=[0.125, 0.25, 0.5, 1., 2.],
    save_path=name_instance('random_seed', 'mode', 'n_samples', 'finetune_scaling', base_folder='data/cifar/main/'),
    load_path=(lambda array_id, random_seed, **kwargs: f'data/cifar/pretrain/random_seed={random_seed}/model.pt'),
    model='resnet'
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
