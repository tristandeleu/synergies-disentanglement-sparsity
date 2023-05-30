import jax.numpy as jnp

from collections import namedtuple
from jax.nn import one_hot
from jaxopt import BlockCoordinateDescent, ProximalGradient
from jaxopt import objective, prox
from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy
from jax_meta.metalearners.base import MetaLearner

from utils.sparsity import (
    prox_group_lasso, prox_group_lasso_intercept, group_support,
    prox_group_svm, block_prox_svm, multiclass_linear_svm_osqp,
    multiclass_linear_svm_primal_osqp,
    multiclass_linear_group_enet_svm_dual)
from utils.sparse_implicit_diff import sparsify


SparseBiLevelState = namedtuple('SparseBiLevelState', ['model', 'log_scale', 'log_plam'])


class SparseBiLevel(MetaLearner):
    def __init__(
            self, encoder, num_ways, maxiter_inner=5, est="sparse_logreg",
            inner_solver='pgd', l1reg=1e-2, tol=1e-3, l2reg=1, unroll=False):
        super().__init__()
        self.encoder = encoder
        self.num_ways = num_ways
        self.maxiter_inner = maxiter_inner
        self.est = est
        self.inner_solver = inner_solver
        self.l1reg = l1reg
        self.l2reg=l2reg
        self.tol = tol
        self.unroll = unroll

    def loss(self, params, log_scale, inputs, targets):
        if isinstance(params, tuple):
            weights, intercept = params[0], params[1]
            logits = jnp.matmul(inputs, weights) + intercept
        else:
            weights = params
            logits = jnp.matmul(inputs, weights)
        logits = logits * jnp.exp(log_scale)
        loss = jnp.mean(cross_entropy(logits, targets))
        logs = {
            'loss': loss,
            'accuracy': accuracy(logits, targets),
            'sparsity': jnp.mean(
                jnp.linalg.norm(weights, ord=2, axis=1) == 0.),
        }
        return loss, ({}, logs)  # No state for the classifier

    def adapt(self, inputs, targets, log_plam):
        if self.est == "sparse_logreg":
            targets_one_hot = one_hot(targets, num_classes=self.num_ways)
            lambda_max = jnp.linalg.norm(
                inputs.T @ targets_one_hot, axis=-1).max() / 4
            lambda_max /= inputs.shape[0]

            if self.inner_solver == 'pgd':
                inner_solver = sparsify(ProximalGradient)(
                    fun=objective.multiclass_logreg,
                    prox=prox_group_lasso,
                    support_fun=group_support,
                    maxiter=self.maxiter_inner,
                    implicit_diff=(not self.unroll),
                    tol=self.tol,
                )
            if self.inner_solver == 'pcd':
                inner_solver = sparsify(BlockCoordinateDescent)(
                    fun=objective.multiclass_logreg,
                    block_prox=prox.prox_group_lasso,
                    support_fun=group_support,
                    maxiter=self.maxiter_inner,
                    implicit_diff=(not self.unroll),
                    tol=self.tol,
                )

            init_params = jnp.zeros((inputs.shape[1], self.num_ways))
            sol = inner_solver.run(
                init_params,
                hyperparams_prox=jnp.exp(self.l1reg) * lambda_max,
                data=(inputs, targets)
            )
            params, logs = sol.params, sol.state._asdict()
            logs['lambda'] = jnp.exp(log_plam) * lambda_max
        if self.est == "sparse_logreg_intercept":
            # TODO bug in jaxopt for blockCD + intercept
            # inner_solver = BlockCoordinateDescent(
            #     fun=objective.multiclass_logreg_with_intercept,
            #     block_prox=block_prox_group_lasso_intercept,
            #     maxiter=self.maxiter_inner,
            #     implicit_diff=(not self.unroll),
            #     tol=self.tol,
            #     # acceleration=False
            # )
            inner_solver = ProximalGradient(
                fun=objective.multiclass_logreg_with_intercept,
                prox=prox_group_lasso_intercept,
                maxiter=self.maxiter_inner,
                implicit_diff=(not self.unroll),
                tol=self.tol,
                acceleration=False
            )
            targets_one_hot = one_hot(targets, num_classes=self.num_ways)
            lambda_max = jnp.linalg.norm(
                inputs.T @ targets_one_hot, axis=-1).max() / 4
            # it is important to normalize by the number of samples
            lambda_max /= inputs.shape[0]

            init_params = (
                jnp.zeros((inputs.shape[1], self.num_ways)),
                jnp.zeros(self.num_ways))
            sol = inner_solver.run(
                init_params,
                hyperparams_prox=jnp.exp(self.l1reg) * lambda_max,
                data=(inputs, targets)
            )
            params, logs = sol.params, sol.state._asdict()
            logs['lambda'] = jnp.exp(log_plam) * lambda_max
        elif self.est == "svm":
            targets_one_hot = one_hot(targets, num_classes=self.num_ways)
            # be careful that optimization is done the dual for SVM
            # hence init_params is of shape (inputs.shape[0], self.num_ways)
            init_params = jnp.zeros((inputs.shape[0], self.num_ways))

            if self.inner_solver in ['pgd', 'pcd']:
                if self.inner_solver == 'pgd':
                    inner_solver = ProximalGradient(
                        fun=objective.multiclass_linear_svm_dual,
                        prox=prox_group_svm,
                        maxiter=self.maxiter_inner,
                        implicit_diff=(not self.unroll),
                        tol=self.tol,
                    )
                elif self.inner_solver == 'pcd':
                    inner_solver = BlockCoordinateDescent(
                        fun=objective.multiclass_linear_svm_dual,
                        block_prox=block_prox_svm,
                        maxiter=self.maxiter_inner,
                        implicit_diff=(not self.unroll),
                        tol=self.tol,
                    )
                sol_dual = inner_solver.run(
                    init_params,
                    hyperparams_prox=None,
                    l2reg=self.l2reg,
                    data=(inputs, targets_one_hot)
                )
                # then one has to go back in the primal
                sol_primal = jnp.dot(
                    inputs.T, (targets_one_hot - sol_dual.params)) / self.l2reg
                params, logs = sol_primal, sol_dual.state._asdict()
            elif self.inner_solver == 'qp':
                sol_dual = multiclass_linear_svm_osqp(
                    inputs, targets_one_hot, self.l2reg, self.tol,
                    self.maxiter_inner)
                sol_primal = jnp.dot(
                    inputs.T, (targets_one_hot - sol_dual)) / self.l2reg
                params, logs = sol_primal, {}
            elif self.inner_solver == 'qp_primal':
                params, state = multiclass_linear_svm_primal_osqp(
                    inputs, targets_one_hot, self.l2reg, self.tol,
                    self.maxiter_inner)
                logs = {}
        elif self.est == "group_enet_svm":
            # Computation to be double checked
            targets_one_hot = one_hot(targets, num_classes=self.num_ways)
            init_params = jnp.zeros((inputs.shape[0], self.num_ways))

            l1reg = self.plam * jnp.sqrt(inputs.shape[1])
            if self.inner_solver == 'pcd':
                inner_solver = BlockCoordinateDescent(
                    fun=multiclass_linear_group_enet_svm_dual,
                    block_prox=block_prox_svm,
                    maxiter=self.maxiter_inner,
                    implicit_diff=(not self.unroll),
                    tol=self.tol,
                )
                sol_dual = inner_solver.run(
                    init_params,
                    l2reg=self.l2reg,
                    l1reg=l1reg,
                    hyperparams_prox=None,
                    data=(inputs, targets_one_hot))
                sol_primal = prox_group_lasso(
                    jnp.dot(inputs.T, (targets_one_hot - sol_dual.params)),
                    l1reg, 1) / self.l2reg
            params, logs = sol_primal, sol_dual.state._asdict()
        # Add loss on the training set to the logs
        log_scale = jnp.array(0.)
        _, (_, inner_logs) = self.loss(params, log_scale, inputs, targets)
        logs.update(inner_logs)

        return (params, logs)

    def outer_loss(self, params, state, train, test, args):
        train_features, _ = self.encoder.apply(
            params.model, state, train.inputs, *args
        )
        adapted_params, inner_logs = self.adapt(
            train_features, train.targets, params.log_plam
        )

        test_features, state = self.encoder.apply(
            params.model, state, test.inputs, *args
        )
        outer_loss, (_, outer_logs) = self.loss(
            adapted_params, params.log_scale, test_features, test.targets
        )
        outer_logs.update({
            'log_scale': params.log_scale,
            'log_plam': params.log_plam
        })
        return outer_loss, state, inner_logs, outer_logs

    def meta_init(self, key, *args, **kwargs):
        model_params, state = self.encoder.init(key, *args, **kwargs)
        params = SparseBiLevelState(
            model=model_params,
            log_scale=jnp.array(0.),
            log_plam=jnp.log(self.plam),
        )
        return (params, state)
