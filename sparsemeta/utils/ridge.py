import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve


def ridge_objective(params, l2reg, data):
  """Ridge objective function."""
  X_tr, y_tr = data
  residuals = jnp.dot(X_tr, params) - y_tr
  return 0.5 * jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.sum(params ** 2)


@implicit_diff.custom_root(jax.grad(ridge_objective))
def ridge_solver(init_params, l2reg, data):
  """Solve ridge regression by conjugate gradient."""
  X_tr, y_tr = data

  def matvec(u):
    return jnp.dot(X_tr.T, jnp.dot(X_tr, u))

  return linear_solve.solve_cg(matvec=matvec,
                               b=jnp.dot(X_tr.T, y_tr),
                               ridge=len(y_tr) * l2reg,
                               init=init_params,
                               maxiter=1000)
