# Synergies between Disentanglement and Sparsity: Generalization and Identifiability in Multi-Task Learning (ICML 2023)

This repository contains the official implementation of ([Lachapelle et al., 2023](https://arxiv.org/abs/2211.14666)).

## Installation

To avoid any conflict with your existing Python setup, it is recommended to work in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

Follow these [instructions](https://github.com/google/jax#installation) to install the version of JAX corresponding to your versions of CUDA and CuDNN.
```bash
pip install -r requirements.txt
pip install -e .
```

## Experiment (3D Shapes)
To reproduce our disentanglement experiment on 3D Shapes (Figure 4), you can run the following script:

```bash
python sparsemeta/main_regression.py \
    --meta_lr 0.001 \
    --num_batches 20000 \
    --rep_norm batch_norm \
    --z_dim 6 \
    --shots 25 \
    --test_shots 25 \
    --use_plam \
    --l1reg 0.3 \
    --outer_l1reg 0.0 \
    --l2reg 1e-07 \
    --outer_l2reg 0.0 \
    --use_ridge_solver \
    --task_mode binomial_gauss \
    --weight_decay 0.0 \
    --maxiter_inner 1000 \
    --inner_solver pcd \
    --dis_eval_every 1000 \
    --no_inner_outer_split \
    --scale_noise 0.1 \
    --z_noise_scale 1.0 \
    --z_dist harder_gauss_0.9 \
    --dataset Regression3DShapes
```

## Citation
If you want to cite our work, please use the following Bibtex entry:

```
@article{lachapelle2023synergiesmultitask,
    title={{Synergies between Disentanglement and Sparsity: Generalization and Identifiability in Multi-Task Learning}},
    author={Lachapelle, Sebastien and Deleu, Tristan and Mahajan, Divyat and Mitliagkas, Ioannis and Bengio, Yoshua, and Lacoste-Julien, Simon and Bertrand, Quentin},
    journal={International Conference on Machine Learning (ICML)},
    year={2023}
}
```