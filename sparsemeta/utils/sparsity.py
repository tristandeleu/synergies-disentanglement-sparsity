from typing import Any
from typing import Optional
from functools import partial

import jax.numpy as jnp
from jax import tree_util as tu
from jax import vmap, jit, tree_util, lax
from jaxopt import prox, projection, OSQP
from jaxopt import tree_util
from jaxopt._src import base

from jaxopt.objective import CompositeLinearFunction


def prox_group_lasso_intercept(
        params: Any,
        hyperparams: Optional[float] = 1.0,
        scaling=1.0) -> Any:
    weights, intercept = params
    _prox_group_lasso = vmap(
      prox.prox_group_lasso, in_axes=(0, None, None))
    return _prox_group_lasso(weights, hyperparams, scaling), intercept


def block_prox_group_lasso_intercept(
        params: Any,
        hyperparams: Optional[float] = 1.0,
        scaling=1.0) -> Any:
    weights, intercept = params
    return prox.prox_group_lasso(weights, hyperparams, scaling), intercept


block_prox_svm = prox.make_prox_from_projection(projection.projection_simplex)

prox_group_svm = vmap(
    prox.make_prox_from_projection(projection.projection_simplex),
    in_axes=(0, None, None))


prox_group_lasso = vmap(prox.prox_group_lasso, in_axes=(0, None, None))


def group_support(params):
    norms = jnp.linalg.norm(params, axis=-1, keepdims=True)
    return (norms > 0)

def support(params):
    return params != 0

def sparsity_patterns(metalearner, params, state, dataset, *args):
    """Average sparsity patterns of the parameters after adaptation
    with proximal gradient descent (with ANILGroupLasso).
    """
    @partial(jit, static_argnums=(0, 4))
    @partial(vmap, in_axes=(None, None, None, 0, None))
    def _sparsity_patterns(metalearner, params, state, train, args):
        features, _ = metalearner.encoder.apply(
            params, state, train.inputs, *args
        )
        adapted_params, _ = metalearner.adapt(features, train.targets)

        pattern = (jnp.linalg.norm(adapted_params, ord=2, axis=1) != 0)
        return pattern.astype(jnp.float32)

    patterns, infos = [], []
    for batch in dataset:
        infos.append(batch['train'].infos)
        patterns.append(_sparsity_patterns(
            metalearner, params, state, batch['train'], args
        ))

    patterns = tu.tree_map(lambda *x: jnp.concatenate(x, axis=0), *patterns)
    infos = tu.tree_map(lambda *x: jnp.concatenate(x, axis=0), *infos)
    return (patterns, infos)


# The following code is taken from
# https://jaxopt.github.io/stable/auto_examples/constrained/multiclass_linear_svm.html#sphx-glr-auto-examples-constrained-multiclass-linear-svm-py
def multiclass_linear_svm_osqp(X, Y, l2reg, tol, maxiter_inner):
  # We solve the problem
  #
  #   minimize 0.5/l2reg beta X X.T beta - (1. - Y)^T beta - 1./l2reg (Y^T X) X^T beta
  #   under        beta >= 0
  #         sum_i beta_i = 1
  #
  def matvec_Q(X, beta):
    return 1./l2reg * jnp.dot(X, jnp.dot(X.T, beta))

  linear_part = - (1. - Y) - 1./l2reg * jnp.dot(X, jnp.dot(X.T, Y))

  def matvec_A(_, beta):
    return jnp.sum(beta, axis=-1)

  def matvec_G(_, beta):
    return -beta

  b = jnp.ones(X.shape[0])
  h = jnp.zeros_like(Y)

  osqp = OSQP(
      matvec_Q=matvec_Q, matvec_A=matvec_A, matvec_G=matvec_G, tol=tol, maxiter=maxiter_inner)
  hyper_params = dict(params_obj=(X, linear_part),
                      params_eq=(None, b),
                      params_ineq=(None, h))
  sol, _ = osqp.run(init_params=None, **hyper_params)
  return sol.primal


def multiclass_linear_svm_primal_osqp(X, Y, l2reg, tol, maxiter_inner):
  # Solve SVM optimization problem in the primal
  def matvec_Q(X, x):
    # x = (x_, W)
    result = jnp.zeros(n_samples + n_classes * n_features)
    result = result.at[-n_classes * n_features:].set(
        l2reg * x[-n_classes * n_features:])
    return result

  n_samples, n_features = X.shape
  n_classes = Y.shape[1]
  linear_part = jnp.zeros(n_samples + n_classes * n_features)
  linear_part = linear_part.at[:n_samples].set(1)

  def matvec_G(X_Y, x):
    X, Y = X_Y
    xi = x[:n_samples]
    W = x[-n_classes * n_features:].reshape((n_features, n_classes))
    result = X @ W
    result = result - (Y * (X @ W)).sum(axis=1, keepdims=True)
    # result -= lax.broadcast(
    #   (Y * (X @ W)).sum(axis=1), [n_classes]).T
    # result -= lax.broadcast(xi, [n_classes]).T
    result = result - jnp.expand_dims(xi, axis=1)
    return result

  h = Y - 1

  osqp = OSQP(
    matvec_Q=matvec_Q, matvec_G=matvec_G, tol=tol, maxiter=maxiter_inner)
  hyper_params = dict(
    params_obj=(X, linear_part), params_ineq=((X, Y), h))

  sol, state = osqp.run(init_params=None, **hyper_params)
  W_osqp_primal = sol.primal[-n_classes * n_features:].reshape((n_features, n_classes))
  return W_osqp_primal, state


class MulticlassLinearGroupEnetSvmDual(CompositeLinearFunction):
  """Dual objective function of multiclass linear SVMs."""

  def subfun(self, Xbeta, l1reg, l2reg, data):
    X, Y = data
    XY = jnp.dot(X.T, Y)

    # The dual objective is:
    # fun(beta) = vdot(beta, 1 - Y) - 0.5 / l2reg * ||V(beta)||^2
    # where V(beta) = dot(X.T, Y) - dot(X.T, beta).
    V = XY - Xbeta
    V = prox_group_lasso(V, l1reg, 1)
    # V = prox.prox_group_lasso(V, l1reg)
    # With opposite sign, as we want to maximize.
    return 0.5 / l2reg * jnp.vdot(V, V)

  def make_linop(self, l1reg, l2reg, data):
    """Creates linear operator."""
    return base.LinearOperator(data[0].T)

  def columnwise_lipschitz_const(self, l1reg, l2reg, data):
    """Column-wise Lipschitz constants."""
    linop = self.make_linop(l1reg, l2reg, data)
    return linop.column_l2_norms(squared=True)

  def b(self, l1reg, l2reg, data):
    return data[1]


multiclass_linear_group_enet_svm_dual = MulticlassLinearGroupEnetSvmDual()
