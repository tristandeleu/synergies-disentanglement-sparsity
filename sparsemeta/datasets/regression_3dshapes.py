from collections import namedtuple
import h5py
import numpy as np

import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.stats import multivariate_normal
from jax_meta.datasets.base import MetaDataset
from jax_meta.utils.data import download_url, get_asset

Dataset = namedtuple('Dataset', ['inputs', 'targets', 'factors', 'infos'])

class Regression3DShapes(MetaDataset):
    name = 'regression3dshapes'
    url = 'https://storage.googleapis.com/3d-shapes'
    filenames = ['3dshapes.h5']
    shape = (64, 64, 3)
    factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    num_values_per_factor = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}

    def __init__(
            self,
            root,
            batch_size,
            shots=5,
            test_shots=None,
            size=None,
            split='train',
            split_mode="same",
            task_mode="laplace",
            seed=0,
            download=False,
            _data=None,
            _factors=None,
            scale_noise=0.1,
            z_dist="uniform",
            z_noise_scale=1.
        ):
        super().__init__(root, batch_size, shots=shots, ways=1,
            test_shots=test_shots, size=size, split=split, seed=seed,
            download=download)
        self.z_dim = len(self.factor_names)
        self.z_noise_scale = z_noise_scale
        if _data is None or _factors is None:
            self.load_data()
        else:
            # This is to avoid loading the dataset twice in RAM
            self._data = _data
            self._factors = _factors
            self.num_images = self._data.shape[0]

        # SL: using different seeds for different splits
        assert split in ["train", "val", "test"]
        if split == "val":
            self.seed = self.seed + 1
        elif split == "test":
            self.seed = self.seed + 2

        self.split_mode = split_mode
        self.task_mode = task_mode
        self.scale_noise = scale_noise
        self.z_dist = z_dist
        if self.z_dist.startswith("gauss"):
            print("Computing pmf of factors...")
            rho = float(self.z_dist.split("_")[1])
            mean = np.zeros((self.z_dim))
            cov = compute_cov(self.z_dim, rho)
            scores = multivariate_normal.pdf(self._factors, mean=mean, cov=cov)
            normalization = scores.sum()
            self.z_probabilities = scores / normalization
            print("Done.")
        elif self.z_dist.startswith("harder_gauss"):
            print("Computing pmf of factors...")
            rho = float(self.z_dist.split("_")[-1])
            mean = np.zeros((self.z_dim))
            cov = compute_cov_v2(self.z_dim, rho)
            scores = multivariate_normal.pdf(self._factors, mean=mean, cov=cov)
            normalization = scores.sum()
            self.z_probabilities = scores / normalization
            print("Done.")
        elif self.z_dist == "uniform":
            self.z_probabilities = np.ones(self.num_images) / self.num_images
        else:
            raise NotImplementedError(f"--z_dist {self.z_dist} is not implemented.")

        # constructing finitely many coefficient vectors
        if self.task_mode.startswith("finite_intra_support"):
            support_size = int(self.task_mode.split("_")[-1])
            self.coefficients = self.rng.normal(size=(self.z_dim * support_size, self.z_dim))
            for i in range(self.z_dim):
                support = np.zeros((support_size, self.z_dim))
                support[:, i: min(self.z_dim, i + support_size)] = 1
                if self.z_dim < i + support_size:
                    support[:, 0: i + support_size - self.z_dim] = 1
                self.coefficients[i * support_size: (i+1) * support_size] = self.coefficients[i * support_size: (i+1) * support_size] * support


    def load_data(self):
        if self._data is None:
            print("Loading dataset in RAM...")
            dataset = h5py.File(self.folder / self.filenames[0], 'r')
            n_samples = dataset["images"].shape[0]
            # set seed for reproducibility
            # rng = check_random_state(0)
            # samples_to_select = rng.choice(
            #     np.arange(n_samples), 1000, replace=False)
            # samples_to_select.sort()
            # "[:]" loads everything in RAM
            self._data = dataset["images"][:]
            # "[:]" loads everything in RAM
            self._factors = dataset["labels"][:]
            self._factors = self._factors.astype(np.float32)

            self.z_values = {}
            for i, name in enumerate(self.factor_names):
                self.z_values[name] = np.unique(self._factors[:, i])

            if self.z_noise_scale != 0.0:
                rng = np.random.default_rng(546243265)

                gaps = {}
                for name in self.factor_names:
                    gaps[name] = self.z_noise_scale * (self.z_values[name][1] - self.z_values[name][0]) / 2

                high = np.array([gaps["floor_hue"], gaps["wall_hue"], gaps["object_hue"], gaps["scale"], gaps["shape"],
                                 gaps["orientation"]])
                low = - high

                self._factors += rng.uniform(low=low, high=high, size=self._factors.shape)

            # standardize factors
            self._factors = (self._factors - self._factors.mean(0)) / self._factors.std(0)

            self.num_images = self._data.shape[0]

            print("Loading completed.")
        return self

    def get_indices(self):
        total_shots = self.shots + self.test_shots
        indices = self.rng.choice(self.num_images, size=(self.batch_size, total_shots), p=self.z_probabilities)
        return indices

    def sample_coefficients(self):
        if self.task_mode == "binomial_gauss":
            support = self.rng.binomial(1, 0.5, size=(self.batch_size, self.z_dim,))
            # making sure support is not empty!!
            rand_idx = self.rng.choice(range(self.z_dim), size=(6,))
            support[:, rand_idx] = 1
            coefficients = self.rng.normal(size=(self.batch_size, self.z_dim)) * support
        elif self.task_mode.startswith("fix_size_support"):
            # sampling supports of size `support_size`
            support_size = int(self.task_mode.split("_")[-1])
            assert 1 <= support_size <= self.z_dim
            support = np.zeros((self.batch_size, self.z_dim))
            support[:, :support_size] = 1
            # TODO: get rid of for loop.
            for i in range(self.batch_size):
                support[i, :] = self.rng.permutation(support[i, :])
            coefficients = self.rng.normal(size=(self.batch_size, self.z_dim)) * support
        elif self.task_mode.startswith("block_support"):
            block_size = int(self.task_mode.split("_")[-1])
            assert self.z_dim % block_size == 0
            num_block = self.z_dim // block_size
            support = np.zeros((self.batch_size, self.z_dim))
            for i in range(self.batch_size):
                rand_idx = self.rng.integers(0, num_block)
                support[i, rand_idx * block_size : (rand_idx + 1) * block_size] = 1
            coefficients = self.rng.normal(size=(self.batch_size, self.z_dim)) * support
        elif self.task_mode == "laplace":
            # Coefficient are Laplacian.
            coefficients = self.rng.laplace(size=(self.batch_size, self.z_dim))
        elif self.task_mode == "gauss":
            # Coefficient are Gaussian.
            coefficients = self.rng.normal(size=(self.batch_size, self.z_dim))
        elif self.task_mode.startswith("finite_intra_support"):
            number_of_tasks = self.coefficients.shape[0]
            task_idx = self.rng.choice(number_of_tasks, size=(self.batch_size,))
            coefficients = self.coefficients[task_idx]
        else:
            raise NotImplementedError(f"--task_mode {self.task_mode} is not implemented.")

        bias = self.rng.normal(size=(self.batch_size,))

        return coefficients, bias

    def compute_targets(self, factors, coefficients, bias):
        num_shots = factors.shape[1]
        noise = self.rng.normal(
            size=(self.batch_size, num_shots), scale=1)
        # import ipdb; ipdb.set_trace()
        # TODO: add bias when model has one.
        # b stands for batch
        # s for shots
        # f for factors
        Zw = np.einsum("bsf,bf->bs", factors, coefficients)
        normalizing_coef = np.linalg.norm(Zw, axis=1) / np.linalg.norm(noise, axis=1)
        # normalizing_coef = np.sqrt((Zw **2).mean(axis=1))
        # / (noise ** 2).mean(axis=1)
        return Zw  + self.scale_noise * normalizing_coef[ :, None] * noise #+ bias[:, None]

    def plot_factors(self):
        fig, axes = plt.subplots(self.z_dim, self.z_dim, sharex=True, sharey=True, figsize=(15, 15))

        # sampling from the joint.
        indices = self.rng.choice(self.num_images, size=(100000,), p=self.z_probabilities)
        sampled_factors = self._factors[indices]

        max_z, min_z = np.max(self._factors), np.min(self._factors)

        for i in range(self.z_dim):
            for j in range(self.z_dim):
                axes[i,j].hist2d(sampled_factors[:,j], sampled_factors[:,i], bins=50, range=[[min_z, max_z],[min_z, max_z]])
                if j == 0:
                    axes[i,j].set_ylabel(self.factor_names[i])
                if i == self.z_dim - 1:
                    axes[i,j].set_xlabel(self.factor_names[j])
        return fig

    def __iter__(self):  # overwriting MetaDataset.__iter__
        shape = self.data.shape[1:]

        # SL: reinit RNG to make sure the dataset is the same every time.
        self.reset()

        while (self.size is None) or (self.num_samples < self.size):
            indices = self.get_indices()
            train_data = self.transform(self.data[indices[..., :self.shots]])
            test_data = self.transform(self.data[indices[..., self.shots:]])

            train_factors = self._factors[indices[..., :self.shots]]
            test_factors = self._factors[indices[..., self.shots:]]

            coefficients, bias = self.sample_coefficients()
            train_targets = self.compute_targets(train_factors, coefficients, bias)
            test_targets = self.compute_targets(test_factors, coefficients, bias)

            train = Dataset(
                inputs=train_data,  # (batch_size, num_shots, 64, 64, 3)
                targets=train_targets,  # (batch_size, num_shots)
                factors=train_factors,  # (batch_size, num_shots, 6)
                infos={'indices': indices[..., :self.shots]}
            )
            test = Dataset(
                inputs=test_data,
                targets=test_targets,
                factors=test_factors,
                infos={'indices': indices[..., self.shots:]}
            )

            self.num_samples += 1
            yield {'train': train, 'test': test}

    def transform(self, data):
        return data.astype(np.float32) / 255.

    def factors2index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
          factors: np array shape [6,batch_size].
                   factors[i]=factors[i,:] takes integer values in
                   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

        Returns:
          indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self.factor_names))):
            indices += factors[factor] * base
            base *= self.num_values_per_factor[name]
        return indices

    def _check_integrity(self):
        if not (self.folder / self.filenames[0]).is_file():
            return False
        return True

    def download(self, max_workers=8):
        if self._check_integrity():
            return
        print("Downloading dataset...")
        download_url(f'{self.url}/{self.filenames[0]}', self.folder,
                     self.filenames[0])
        print("Download completed.")


def compute_cov(z_dim, rho):
    assert 0 <= rho <= 1
    cov = np.zeros((z_dim, z_dim))
    for i in range(z_dim):
        for j in range(z_dim):
            if i == j:
                cov[i,j] = 1.
            else:
                cov[i,j] = rho ** np.abs(i - j)
    return cov

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
