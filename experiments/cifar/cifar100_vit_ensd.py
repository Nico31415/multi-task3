import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(50)),
    aux_n_samples=[1000],#[10, 20, 50, 100, 200, 500, 1000],
    scaling=[0.125, 0.25, 0.5, 1., 2.],
    save_path=name_instance('random_seed', 'n_samples', 'scaling', base_folder='data/cifar_vit/ensd/'),
    load_path_pre=(lambda array_id, random_seed, **kwargs: f'data/cifar_vit/pretrain/random_seed={random_seed}/model.pt'),
    load_path_post=(lambda array_id, random_seed, n_samples, scaling, **kwargs: f'data/cifar_vit/main/random_seed={random_seed}--mode=finetuning--n_samples={n_samples}--finetune_scaling={scaling}/model.pt'),
    model='vit',
    device='cpu',
    batch_size=4000
)

def main(args):
    argparse_array.call_script('experiments/cifar/ensd.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
