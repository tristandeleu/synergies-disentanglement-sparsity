import numpy as np
import os
import h5py
import wandb
import haiku as hk
import jax
import jax.numpy as jnp
from collections import namedtuple
from scipy.stats import multivariate_normal
from sklearn.decomposition import FastICA
import time

from jax_meta.utils import io

from sparsemeta.disentanglement_metrics import get_z_z_hat_uniform, get_z_z_hat, disentanglement, completeness
from sparsemeta.conv import Conv4, ConvDisentanglement
from sparsemeta.datasets.regression_3dshapes import Regression3DShapes


def load_3d_shapes():
    dataset = Regression3DShapes(
        root=os.getenv('SLURM_TMPDIR'),
        batch_size=1,
        split="test",
        z_noise_scale=1.0,  # QKFIX: all the runs we are interested in have z_noise_scale = 1
        z_dist="harder_gauss_0.9",  # This will be updated for every run.
        download=True
    )

    return dataset

def compute_cov_v2(z_dim, rho):
    assert 0 <= rho <= 1
    cov = np.zeros((z_dim, z_dim))
    for i in range(z_dim):
        for j in range(z_dim):
            if i == j:
                cov[i,j] = 1
            else:
                cov[i,j] = rho
    return cov

def make_encoder(config):
    # Adapted from sparsemeta/main_regression.py
    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder(inputs, is_training):
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

    return encoder


def main(args):
    # Get run from wandb
    api = wandb.Api()
    entity, project = 'utimateteam', 'sparse-meta-learning'
    runs = api.runs(entity + "/" + project, filters={"tags": {"$in": ["oct7_varying_corr"]}})

    # Load 3D-Shapes dataset
    dataset = load_3d_shapes()

    for i, run in enumerate(runs):
        print("run #", i, "/", len(runs))
        print(run.url)
        #print(run.config.keys())
        if args.zero_l1 and run.config["solver"]["l1reg"] != 0:
            continue
        elif not args.zero_l1 and run.config["solver"]["l1reg"] == 0:
            continue
        t0 = time.time()

        # Setting the z_dist option of the dataset to that of the current run
        dataset.z_dist = run.config["z_dist"]
        assert dataset.z_dist.startswith("harder_gauss")
        assert run.config["z_noise_scale"] == 1
        rho = float(dataset.z_dist.split("_")[-1])
        mean = np.zeros((dataset.z_dim))
        cov = compute_cov_v2(dataset.z_dim, rho)
        scores = multivariate_normal.pdf(dataset._factors, mean=mean, cov=cov)
        normalization = scores.sum()
        dataset.z_probabilities = scores / normalization  # this z_probabilities will be used in get_z_z_hat

        # Create the encoder
        encoder = make_encoder(run.config)

        # Download last model
        model_name = 'model.npz'
        run.file(model_name).download(root=os.getenv('SLURM_TMPDIR'), replace=True)
        with open(os.path.join(os.getenv('SLURM_TMPDIR'), model_name), 'rb') as f:
            best_model = io.load(f)
            params = best_model['params']
            state = best_model.get('state', {})  # QKFIX: If the state is empty

        # getting z and z_hat
        State = namedtuple('State', ['model'])
        state_hack = State(model=state)
        z, z_hat = get_z_z_hat_uniform(dataset, encoder, params, state_hack)

        # ICA representation
        _, z_hat_train = get_z_z_hat(dataset, encoder, params, state_hack, num_samples=250000)
        algo = FastICA(n_components=z_hat.shape[1], whiten="unit-variance", fun='logcosh')
        algo = algo.fit(z_hat_train)  # fit ICA on "in-distribution" z_hat
        z_hat_ica = algo.transform(z_hat)  # Transform the "uniform" z_hat

        # standardizing representations
        z = (z - np.mean(z, 0)) / np.std(z, 0)
        z_hat = (z_hat - np.mean(z_hat, 0)) / np.std(z_hat, 0)
        z_hat_ica = (z_hat_ica - np.mean(z_hat_ica, 0)) / np.std(z_hat_ica, 0)

        run.summary["dci_d"] = disentanglement(z_hat, z)
        run.summary["dci_c"] = completeness(z_hat, z)
        run.summary["dci_d_ica"] = disentanglement(z_hat_ica, z)
        run.summary["dci_c_ica"] = completeness(z_hat_ica, z)
        #run.summary["dci_d_lasso"] = disentanglement(z_hat, z, lasso=True)
        #run.summary["dci_c_lasso"] = completeness(z_hat, z, lasso=True)
        run.summary.update()
        print("    MCC = ", run.summary["train/dis/mcc"])
        print("    D = ", run.summary["dci_d"])
        print("    C = ", run.summary["dci_c"])
        print("    D (ICA) = ", run.summary["dci_d_ica"])
        print("    C (ICA) = ", run.summary["dci_c_ica"])
        print("    time = ", (time.time() - t0) / 60., "minutes")
        #print("    D (lasso) = ", run.summary["dci_d_lasso"])
        #print("    C (lasso)= ", run.summary["dci_c_lasso"])

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--zero_l1', action="store_true")
    args = parser.parse_args()
    main(args)