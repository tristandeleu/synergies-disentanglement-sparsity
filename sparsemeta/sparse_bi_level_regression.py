import numpy as np
import jax
import jax.numpy as jnp

from jax.nn import one_hot
from jaxopt import BlockCoordinateDescent, ProximalGradient
from jaxopt import objective
from jaxopt import prox
from jax_meta.metalearners.base import MetaLearner

from utils.sparsity import support
from utils.sparse_implicit_diff import sparsify
from utils.ridge import ridge_solver


def grad_safe_abs(x):
    # Default behavior of Jax is that the derivative of the absolute value at zero is one.
    # This function makes sure the derivative at zero is zero.
    return jnp.abs(x) * (x != 0)


class SparseBiLevelRegression(MetaLearner):
    def __init__(
            self, encoder, maxiter_inner=5, l1reg=1e-2, outer_l1reg=0, l2reg=1e-7, outer_l2reg=0.,
            use_ridge_solver=True, tol=1e-3, unroll=False, inner_solver='pgd', use_plam=True, inner_outer_split=True):
        super().__init__()
        self.encoder = encoder
        self.maxiter_inner = maxiter_inner
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.outer_l1reg = outer_l1reg
        self.outer_l2reg = outer_l2reg
        self.use_ridge_solver = use_ridge_solver
        self.tol = tol
        self.unroll = unroll
        self.inner_solver = inner_solver
        self.use_plam = use_plam
        self.inner_outer_split = inner_outer_split

    def get_lambda_l1(self, l1reg, inputs, targets):
        if self.use_plam:
            lambda_max = jnp.abs(inputs.T @ targets).max()
            lambda_max /= inputs.shape[0]
            lambda_max = jax.lax.stop_gradient(lambda_max)

            return l1reg * lambda_max
        else:
            # WARNING: if number of shots differs in inner and outer problems,
            # the value of lambda might differ between inner and outer.
            return l1reg / inputs.shape[0]

    def get_lambda_l2(self, l2reg, inputs):
        if self.use_plam:
            lambda_max = jnp.linalg.norm(inputs) ** 2 / inputs.shape[0]
            lambda_max = jax.lax.stop_gradient(lambda_max)
            return l2reg * lambda_max
        else:
            return l2reg / inputs.shape[0]

    def loss(self, params, inputs, targets):
        y_pred = jnp.matmul(inputs, params)
        loss = 0.5 * jnp.mean((y_pred - targets) ** 2)
        if self.outer_l1reg != 0:
            l1reg = self.get_lambda_l1(self.outer_l1reg, inputs, targets)
            loss += l1reg * grad_safe_abs(params).sum()
        if self.outer_l2reg != 0:
            l2reg = self.get_lambda_l2(self.outer_l2reg, inputs)
            loss += 0.5 * l2reg * (params ** 2).sum()

        logs = {
            'loss': loss,
            'sparsity': params == 0.,  # want the whole vector
        }
        return loss, ({}, logs)  # No state for the classifier

    def adapt(self, inputs, targets):
        init_params = jnp.zeros((inputs.shape[1],))
        if self.l1reg == 0 and self.use_ridge_solver:
            l2reg = self.get_lambda_l2(self.l2reg, inputs)
            params = ridge_solver(init_params, l2reg, (inputs, targets))
            # init_params, self.l2reg / targets.shape[0], (inputs, targets))
            logs = {}
        else:
            if self.inner_solver == 'pgd':
                inner_solver = sparsify(ProximalGradient)(
                    fun=objective.least_squares,
                    prox=prox.prox_lasso,
                    support_fun=support,
                    maxiter=self.maxiter_inner,
                    implicit_diff=(not self.unroll),
                    tol=self.tol,
                )
            if self.inner_solver == 'pcd':
                inner_solver = sparsify(BlockCoordinateDescent)(
                    fun=objective.least_squares,
                    block_prox=prox.prox_lasso,
                    support_fun=support,
                    maxiter=self.maxiter_inner,
                    implicit_diff=(not self.unroll),
                    tol=self.tol,
                )

            l1reg = self.get_lambda_l1(self.l1reg, inputs, targets)  # implements lambda_max trick

            sol = inner_solver.run(
                init_params,
                hyperparams_prox=l1reg,
                data=(inputs, targets)
            )
            params, logs = (sol.params, sol.state._asdict())

        # Add loss on the training set to the logs
        _, (_, inner_logs) = self.loss(params, inputs, targets)
        logs.update(inner_logs)

        return (params, logs)

    def outer_loss(self, params, state, train, test, args):
        if self.inner_outer_split:
            train_features, _ = self.encoder.apply(
                params, state, train.inputs, *args
            )
            adapted_params, inner_logs = self.adapt(train_features, train.targets)

            test_features, state = self.encoder.apply(
                params, state, test.inputs, *args
            )
            outer_loss, (_, outer_logs) = self.loss(
                adapted_params, test_features, test.targets
            )
        else:
            inputs = jnp.concatenate([train.inputs, test.inputs], 0)
            targets = jnp.concatenate([train.targets, test.targets], 0)

            features, state = self.encoder.apply(
                params, state, inputs, *args
            )
            adapted_params, inner_logs = self.adapt(features, targets)

            outer_loss, (_, outer_logs) = self.loss(
                adapted_params, features, targets
            )
        return outer_loss, state, inner_logs, outer_logs


    def meta_init(self, key, *args, **kwargs):
        return self.encoder.init(key, *args, **kwargs)
