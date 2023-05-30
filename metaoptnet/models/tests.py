# # Test for the SVM
# from classification_heads import SparseMetaOptNetHead_SVM_dual


import numpy as np
import torch
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from utils import inner_step_pgd_, inner_step_pcd_
from lightning.classification import SDCAClassifier

iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
n_classes = 3

X = iris.data
X /= np.linalg.norm(X, axis=0)
y = iris.target
n_samples, n_features = X.shape

enc = OneHotEncoder()
targets_one_hot = enc.fit_transform(y.reshape(-1, 1))
targets_one_hot = targets_one_hot.todense()

params = np.ones((n_samples, n_classes)) / n_classes
lambda2 = 100.
stepsizes = lambda2 / np.linalg.norm(X, ord=2) ** 2

###################################################
# PGD
##################################################
params_dual_t = torch.from_numpy(params)
X_t = torch.from_numpy(X)
targets_one_hot_t = torch.from_numpy(targets_one_hot)

for _ in range(1_000):
    params_dual_t  = inner_step_pgd_(
        params_dual_t, X_t, targets_one_hot_t, stepsizes, 0, lambda2)

params_primal_t = X_t.mT @ (targets_one_hot_t- params_dual_t) / lambda2


###################################################
# PCD
##################################################
params_dual_t = torch.from_numpy(params)
X_t = torch.from_numpy(X)
X_t_params = X_t.mT @ params_dual_t
targets_one_hot_t = torch.from_numpy(targets_one_hot)
X_t_target_one_hot = X_t.mT @ targets_one_hot_t

stepsizes_pcd = lambda2 / torch.linalg.norm(X_t, axis=1) ** 2

for _ in range(100):
    params_dual_t, X_t_params = inner_step_pcd_(
        params_dual_t, X_t, X_t_params, X_t_target_one_hot, targets_one_hot_t, stepsizes_pcd, lambda2, 0.)
    # X_t_params = X_t.mT @ params_dual_t

params_primal_t_pcd = X_t.mT @ (targets_one_hot_t- params_dual_t) / lambda2

print(params_primal_t)
print(params_primal_t_pcd)


# 1 / 0

# def test_pgd_vs_jax():
import jax.numpy as jnp
from jaxopt import BlockCoordinateDescent
from jaxopt import OSQP
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox

params_dual_j = jnp.array(params)
X_j = jnp.array(X)
targets_one_hot_j = jnp.array(targets_one_hot)

def multiclass_linear_svm_bcd(X, Y, l2reg):
  print("Block coordinate descent solution:")

  # Set up parameters.
  block_prox = prox.make_prox_from_projection(projection.projection_simplex)
  fun = objective.multiclass_linear_svm_dual
#   data =
  beta_init = jnp.ones((X.shape[0], Y.shape[-1])) / Y.shape[-1]

  # Run solver.
  bcd = BlockCoordinateDescent(
    fun=fun, block_prox=block_prox, maxiter=10*1000, tol=1e-16)
  sol = bcd.run(
    beta_init, hyperparams_prox=None, l2reg=lambda2,
    data=(X, targets_one_hot_j))
  return sol.params



W_bcd = multiclass_linear_svm_bcd(X_j, targets_one_hot_j, lambda2)
W_fit_bcd  = jnp.dot(X_j.T, (targets_one_hot_j - W_bcd)) / lambda2
print(W_fit_bcd)

np.testing.assert_allclose(np.array(W_fit_bcd), params_primal_t.cpu().detach().numpy(), atol=1e-4)

np.testing.assert_allclose(np.array(W_fit_bcd), params_primal_t_pcd.cpu().detach().numpy(), atol=1e-4)
