import numpy as np
import jax.numpy as jnp
import jax
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import scipy
import copy
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV


def get_z_z_hat(dataset, encoder, params, state, num_samples=int(1e6)):
    """Samples are taken from the distribution over Z. Used only to fit ICA."""
    print("Number of images in dataset = ", dataset.num_images)
    z_hat_list = []
    z_list = []
    sample_counter = 0
    i_batch = 0
    batch_size = 256
    while sample_counter < num_samples:
        indices = dataset.rng.choice(dataset.num_images, size=(batch_size,), p=dataset.z_probabilities)
        z_list.append(dataset._factors[indices])
        inputs = dataset._data[indices]
        inputs = dataset.transform(inputs)  # to float32 and rescaling.
        inputs = jax.device_put(inputs)  # send to GPU
        z_hat, _ = encoder.apply(params, state.model, inputs, False)
        z_hat_list.append(z_hat)
        sample_counter += inputs.shape[0]
        i_batch += 1

    z_hat = jnp.concatenate(z_hat_list, 0)[:int(num_samples)]
    z = jnp.concatenate(z_list, 0)[:int(num_samples)]

    return z, z_hat


def get_z_z_hat_uniform(dataset, encoder, params, state, num_samples=int(1e6)):
    """Samples are taken uniformly across the support of Z."""
    if num_samples < dataset._data.shape[0]:
        print(f"Not using all data points to evaluate disentanglement. "
              f"Total data points = {dataset._data.shape[0]}, "
              f"Used data points = {num_samples}.")

    num_samples = min(num_samples, dataset._data.shape[0])

    z_hat_list = []
    sample_counter = 0
    i_batch = 0
    batch_size = 256
    while sample_counter < num_samples:
        inputs = dataset._data[i_batch * batch_size: (i_batch + 1) * batch_size]
        inputs = dataset.transform(inputs)  # to float32 and rescaling.
        inputs = jax.device_put(inputs)  # send to GPU
        z_hat, _ = encoder.apply(params, state.model, inputs, False)
        z_hat_list.append(z_hat)
        sample_counter += inputs.shape[0]
        i_batch += 1

    z_hat = jnp.concatenate(z_hat_list, 0)[:int(num_samples)]
    z = dataset._factors[:int(num_samples)]

    return z, z_hat


def mean_corr_coef_np(z, z_hat, method='pearson', indices=None):
    """
    Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py

    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    x, y = z, z_hat
    d = x.shape[1]
    d_learned = y.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]  # z x z_hat
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))

    cc = np.abs(cc)

    # NaN checker for debugging purposes
    if np.isnan(cc).any():
        print("Found NaN in `cc_program`:")
        print("cc_program = ", cc)
        print("isnan(z_hat) = ", np.isnan(z_hat))
        print("isnan(z) = ", np.isnan(z))
        print("std(z_hat) = ", np.std(z_hat, 0))
        print("std(z) = ", np.std(z, 0))

    assignments = linear_sum_assignment(-1 * cc)
    score = cc[assignments].mean()

    #perm_mat = np.zeros((d, d))
    #perm_mat[assignments] = 1
    # cc_program_perm = np.matmul(perm_mat.transpose(), cc_program)
    #cc_program_perm = np.matmul(cc_program, perm_mat.transpose())  # permute the learned latents

    return score  #, cc_program_perm, assignments


def get_linear_score(x, y):
    reg = LinearRegression(fit_intercept=True).fit(x, y)
    y_pred = reg.predict(x)
    r2s = sklearn.metrics.r2_score(y, y_pred, multioutput='raw_values')
    # when R is very small, we sometimes get very small negative values which cause a NaN below
    r2s = r2s * (1 - (-1e-6 < r2s) * (r2s < 0))
    r = np.mean(np.sqrt(r2s))  # To be comparable to MCC (this is the average of R = coefficient of multiple correlation)
    return r, reg.coef_


def linear_regression_metric(z, z_hat, indices=None):
    # standardize z and z_hat
    z = (z - np.mean(z, 0)) / np.std(z, 0)
    z_hat = (z_hat - np.mean(z_hat, 0)) / np.std(z_hat, 0)

    score, L_hat = get_linear_score(z_hat, z)

    # masking z_hat
    # TODO: this does not take into account case where z_block_size > 1
    if indices is not None:
        z_hat_m = z_hat[:, indices[-z.shape[0]:]]
        score_m, _ = get_linear_score(z_hat_m, z)
    else:
        score_m = 0

    return score, score_m, L_hat


def evaluate_disentanglement(dataset, encoder, params, state, sparsity=None, ica=False):
    """Evaluates out of distribution (uniform latents)"""
    z, z_hat = get_z_z_hat_uniform(dataset, encoder, params, state)

    # Evaluate MCC with full representation (i.e. cheating)
    mcc_full = mean_corr_coef_np(z, z_hat)
    r_full, _, _ = linear_regression_metric(z, z_hat)

    # Evaluate MCC with latents which are used more often
    relevant_factors = sparsity.argsort()[:dataset.z_dim]
    relevant_z_hat = z_hat[:, relevant_factors]
    mcc = mean_corr_coef_np(z, relevant_z_hat)
    r, _, _ = linear_regression_metric(z, relevant_z_hat)

    if ica:
        _, z_hat_train = get_z_z_hat(dataset, encoder, params, state)
        algo = FastICA(n_components=z_hat.shape[1], whiten="unit-variance", fun='logcosh')
        algo = algo.fit(z_hat_train)  # fit ICA on "in-distribution" z_hat
        z_hat_ica = algo.transform(z_hat)  # Transform the "uniform" z_hat
        mcc_full_ica = mean_corr_coef_np(z, z_hat_ica)
        r_full_ica, _, _ = linear_regression_metric(z, z_hat_ica)
        return mcc_full, r_full, mcc, r, mcc_full_ica, r_full_ica, z, z_hat

    return mcc_full, r_full, mcc, r, z, z_hat


def evaluate_disentanglement_old(dataset, encoder, params, state, sparsity=None, ica=False):
    """Evaluate in-distribution"""
    z, z_hat = get_z_z_hat(dataset, encoder, params, state)

    # Evaluate MCC with full representation (i.e. cheating)
    mcc_full = mean_corr_coef_np(z, z_hat)
    r_full, _, _ = linear_regression_metric(z, z_hat)

    # Evaluate MCC with latents which are used more often
    relevant_factors = sparsity.argsort()[:dataset.z_dim]
    relevant_z_hat = z_hat[:, relevant_factors]
    mcc = mean_corr_coef_np(z, relevant_z_hat)
    r, _, _ = linear_regression_metric(z, relevant_z_hat)

    if ica:
        algo = FastICA(n_components=z_hat.shape[1], whiten="unit-variance", fun='logcosh')
        algo = algo.fit(z_hat)  # fit ICA on "in-distribution" z_hat
        z_hat_ica = algo.transform(z_hat)  # Transform the "uniform" z_hat
        mcc_full_ica = mean_corr_coef_np(z, z_hat_ica)
        r_full_ica, _, _ = linear_regression_metric(z, z_hat_ica)
        return mcc_full, r_full, mcc, r, mcc_full_ica, r_full_ica, z, z_hat

    return mcc_full, r_full, mcc, r, z, z_hat


def compute_importance_matrix(z_pred, z, case='disentanglement', lasso=False, fit_intercept=True):
    true_latent_dim = z.shape[1]
    pred_latent_dim = z_pred.shape[1]
    imp_matrix = np.zeros((pred_latent_dim, true_latent_dim))
    for idx in range(true_latent_dim):
        if lasso:
            model = LassoCV(fit_intercept=fit_intercept, cv=3, tol=1e-10).fit(z_pred, z[:, idx])
        else:
            model = LinearRegression(fit_intercept=fit_intercept).fit(z_pred, z[:, idx])
        imp_matrix[:, idx] = model.coef_

    # Taking the absolute value for weights to encode relative importance properly
    imp_matrix = np.abs(imp_matrix)
    if case == 'disentanglement':
        imp_matrix = imp_matrix / np.reshape(np.sum(imp_matrix, axis=1), (pred_latent_dim, 1))
    elif case == 'completeness':
        imp_matrix = imp_matrix / np.reshape(np.sum(imp_matrix, axis=0), (1, true_latent_dim))

    return imp_matrix


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(z_pred, z, lasso=False):
    """Compute the disentanglement score of the representation."""
    importance_matrix = compute_importance_matrix(z_pred, z, case='disentanglement', lasso=lasso, fit_intercept=True)
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(z_pred, z, lasso=False):
    """"Compute completeness of the representation."""
    importance_matrix = compute_importance_matrix(z_pred, z, case='completeness', lasso=lasso, fit_intercept=True)
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)