## Codebase for "Implicit regularization of multi-task learning and finetuning: multiple regimes of feature reuse"

The conda environment is specified in 'environment.yml'.

For each subset, experiments should be run in the following order. Note that each experiment has a number of iterations (which we ran on a computational cluster). Each experiment can be executed by running 'python <filename> <array_id>'. We additionally provide bash scripts that can be used to run these files on a SLURM cluster (though we note that they may require some modifications to run successfully on a cluster different from ours).

- Diagonal linear networks
  - Pretraining (for PT+FT): diagonal_network_pretrain_1.py
  - Overlaps:
    - diagonal_network_overlap_1.py
    - diagonal_network_overlap_2.py
  - Nested feature selection
    - diagonal_overlap_scale.py
    - diagonal_sparse_overlap_1.py
    - diagonal_sparse_overlap_2.py
- ReLU networks
  - Pretraining: relu_scaling_law_2.py
  - Overlap/correlation:
    - relu_finetune_1.py
    - relu_finetune_1b.py
  - Nested feature selection
    - relu_finetune_2.py
    - relu_finetune_2b.py
  - Correlation/magnitude
    - relu_corr.py
    - relu_corr_2.py
- CIFAR-100´
  - ResNet
    - Pretraining: cifar100_pretrain.py
    - Finetuning:
      - cifar100_main.py
      - cifar100_main_2.py
    - ENSD: cifar100_ensd.py
  - ViT:
    - Pretraining: cifar100_pretrain_vit.py
    - Finetuning:
      - cifar100_main_vit.py
      - cifar100_main_2_vit.py
    - ENSD: cifar100_vit_ensd.py

We then collated the resulting files and provide them in the 'data/processed' folder. The different Rmarkdown documents detail how we constructed the figures from there.
