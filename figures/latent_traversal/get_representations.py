import numpy as np
import os
import h5py
import annoy
import wandb
import haiku as hk
import jax
import jax.numpy as jnp

from jax_meta.utils import io

from sparsemeta.conv import Conv4, ConvDisentanglement
from sparsemeta.datasets.regression_3dshapes import Regression3DShapes


def load_3d_shapes():
    dataset = Regression3DShapes(
        root=os.getenv('SLURM_TMPDIR'),
        batch_size=1,
        download=True
    )

    # QKFIX: Avoid postprocessing the factors (normalization)
    h5_dataset = h5py.File(dataset.folder / dataset.filenames[0], 'r')
    dataset._factors = h5_dataset['labels'][:]

    return dataset


def make_encoder(config):
    # Adapted from sparsemeta/main_regression.py
    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder(inputs, is_training):
        features = Conv4(num_filters=config['num_filters'], norm=config['conv_norm'])(inputs, is_training)
        if config['z_dim'] is not None:
            features = jax.nn.relu(hk.Linear(256)(features))
            if config['rep_norm'] == "special_layer_norm":
                features = hk.Linear(config['z_dim'] + 2)(features)  # add two dummy dimensions
            elif config['rep_norm'] == "utimate_norm":
                features = hk.Linear(config['z_dim'] + 10)(features)  # add 10 dummy dimensions
            else:
                features = hk.Linear(config['z_dim'])(features)

        if config['rep_norm'] == "layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features)
        elif config['rep_norm'] == "special_layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features)[:, :-2]  # remove two dummy dimensions
        elif config['rep_norm'] == "utimate_norm":
            features = (features / jnp.linalg.norm(features, ord=2, axis=1, keepdims=True))  # project on 2D sphere
            features = features[:, :-10]  # remove dummy dimensions.
        elif config['rep_norm'] == "batch_norm":
            features = hk.BatchNorm(
                decay_rate=0.9,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features, is_training=is_training)
        elif config['rep_norm'] is not None:
            raise NotImplementedError(f"--rep_norm {config['rep_norm']} is not implemented.")
        return features

    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder_disentanglement(inputs, is_training):
        # when using special_layer_norm, should add two dummy variables...
        outdim = config['z_dim'] + 2 if config['rep_norm'] == "special_layer_norm" else config['z_dim']
        features = ConvDisentanglement(outdim)(inputs, is_training)
        if config['rep_norm'] == "layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features)
        elif config['rep_norm'] == "special_layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features)[:, :-2]  # remove two dummy dimensions
        elif config['rep_norm'] == "batch_norm":
            features = hk.BatchNorm(
                decay_rate=0.9,
                create_scale=config['learn_rep_norm'],
                create_offset=config['learn_rep_norm']
            )(features, is_training=is_training)
        elif config['rep_norm'] is not None:
            raise NotImplementedError(f"--rep_norm {config['rep_norm']} is not implemented.")
        return features

    return encoder_disentanglement if config['encoder'] == 'encoder_disentanglement' else encoder


def main(args):
    # Get run from wandb
    api = wandb.Api()
    run = api.run(args.run_path)

    # Create the encoder
    encoder = make_encoder(run.config)

    # Download best model
    model_name = 'model.npz' if args.last_model else 'best_model.npz'
    run.file(model_name).download(root=os.getenv('SLURM_TMPDIR'), replace=True)
    with open(os.path.join(os.getenv('SLURM_TMPDIR'), model_name), 'rb') as f:
        best_model = io.load(f)
        params = best_model['params']
        state = best_model.get('state', {})  # QKFIX: If the state is empty

    # Load 3D-Shapes dataset
    dataset = load_3d_shapes()

    if args.factor not in dataset.factor_names + ['all']:
        raise ValueError(f'Unknown factor: {args.factor}')

    # Create a NN index for the ground-truth factors
    factor_nn_index = annoy.AnnoyIndex(len(dataset.factor_names), 'euclidean')
    for i, factor in enumerate(dataset._factors):
        factor_nn_index.add_item(i, factor)
    factor_nn_index.build(10)

    # Find the index of the anchor image
    anchor = np.asarray(args.anchor)
    anchor_index, = factor_nn_index.get_nns_by_vector(anchor, 1)
    anchor = dataset._factors[anchor_index]

    factor_names = dataset.factor_names if args.factor == 'all' else [args.factor]

    for factor_name in factor_names:
        # Factor of variation
        factor_index = dataset.factor_names.index(factor_name)
        factor_values = np.unique(dataset._factors[:, factor_index])
        anchor_index, = np.where(factor_values == anchor[factor_index])

        # Create all the factor, by varying the factor of variation
        num_variations = dataset.num_values_per_factor[factor_name]
        if len(factor_values) != num_variations:
            raise ValueError(f'Different number of variations: '
                f'{len(factor_values)} vs {num_variations}')
        factors = np.tile(anchor.copy(), (num_variations, 1))
        factors[:, factor_index] = factor_values

        # Find the indices of the images
        image_indices = []
        for factor in factors:
            image_index, = factor_nn_index.get_nns_by_vector(factor, 1)
            image_indices.append(image_index)
        image_indices = np.asarray(image_indices)

        # Get the latent representation
        raw_images = dataset._data[image_indices]
        images = dataset.transform(raw_images)
        representations, _ = encoder.apply(params, state, images, False)
        
        # Save the results
        with open(args.output_folder / f'latent_{factor_name}_{run.id}.npz', 'wb') as f:
            np.savez(f, factor=factor_name, factors=factors,
                anchor_index=anchor_index, images=raw_images,
                representations=representations, run_id=run.id)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description='Latent traversal in 3D-Shapes')
    parser.add_argument('run_path', type=str, help='Path to the wandb run')
    parser.add_argument('--anchor', type=float, nargs='+', default=[0, 0, 0, 0, 0, 0],
        help='Anchor factors')
    parser.add_argument('--factor', type=str, default='orientation',
        choices=['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation', 'all'],
        help='Factor of variation')
    parser.add_argument('--last_model', action='store_true',
        help='Use the last model (uses "best_model" by default).')
    parser.add_argument('--output_folder', type=Path, default='.',
        help='Output folder for the results.')

    args = parser.parse_args()

    main(args)
