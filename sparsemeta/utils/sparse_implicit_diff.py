import inspect
import jax

from jaxopt import linear_solve
from jaxopt._src.implicit_diff import _extract_kwargs, _signature_bind_and_match, _signature_bind
from jaxopt._src.tree_util import tree_scalar_mul, tree_mul


def sparsify(class_):
    """Sparsify a solver from jaxopt."""
    class _Sparse(class_):
        def __init__(self, *args, support_fun=None, **kwargs):
            super().__init__(*args, **kwargs)
            if support_fun is None:
                raise ValueError('You must specify `support_fun` when you '
                                 'sparsify a solver.')
            self.support_fun = support_fun

        def run(self, init_params, *args, **kwargs):
            run = self._run

            if getattr(self, 'implicit_diff', True):
                reference_signature = getattr(self, 'reference_signature', None)
                decorator = sparse_custom_root(
                    self.optimality_fun,
                    support_fun=self.support_fun,
                    has_aux=True,
                    solve=self.implicit_diff_solve,
                    reference_signature=reference_signature
                )
                run = decorator(run)

            return run(init_params, *args, **kwargs)
    return _Sparse


def sparse_custom_root(
        optimality_fun,
        support_fun,
        has_aux=False,
        solve=linear_solve.solve_normal_cg,
        reference_signature=None
    ):
    if solve is None:
        solve = linear_solve.solve_normal_cg

    def wrapper(solver_fun):
        return _sparse_custom_root(
            solver_fun,
            optimality_fun,
            support_fun,
            solve,
            has_aux,
            reference_signature
        )

    return wrapper


def _sparse_custom_root(
        solver_fun,
        optimality_fun,
        support_fun,
        solve,
        has_aux,
        reference_signature=None
    ):
    solver_fun_signature = inspect.signature(solver_fun)

    if reference_signature is None:
        reference_signature = inspect.signature(optimality_fun)

    elif not isinstance(reference_signature, inspect.Signature):
        fun = getattr(reference_signature, 'subfun', reference_signature)
        reference_signature = inspect.signature(fun)

    def make_custom_vjp_solver_fun(solver_fun, kwarg_keys):
        @jax.custom_vjp
        def solver_fun_flat(*flat_args):
            args, kwargs = _extract_kwargs(kwarg_keys, flat_args)
            return solver_fun(*args, **kwargs)

        def solver_fun_fwd(*flat_args):
            res = solver_fun_flat(*flat_args)
            return res, (res, flat_args)

        def solver_fun_bwd(tup, cotangent):
            res, flat_args = tup
            args, kwargs = _extract_kwargs(kwarg_keys, flat_args)

            if has_aux:
                cotangent = cotangent[0]
                sol = res[0]
            else:
                sol = res

            ba_args, ba_kwargs, map_back = _signature_bind_and_match(
                reference_signature, *args, **kwargs)
            if ba_kwargs:
                raise TypeError(
                    'keyword arguments to solver_fun could not be resolved to '
                    'positional arguments based on the signature '
                    f'{reference_signature}. This can happen under custom_root if '
                    'optimality_fun takes catch-all **kwargs, or under '
                    'custom_fixed_point if fixed_point_fun takes catch-all **kwargs, '
                    'both of which are currently unsupported.')

            vjps = sparse_root_vjp(
                optimality_fun=optimality_fun,
                support_fun=support_fun,
                sol=sol,
                args=ba_args[1:],
                cotangent=cotangent,
                solve=solve
            )
            # Prepend None as the vjp for init_params
            vjps = (None,) + vjps

            arg_vjps, kws_vjps = map_back(vjps)
            ordered_vjps = tuple(arg_vjps) + tuple(kws_vjps[k] for k in kwargs.keys())
            return ordered_vjps

        solver_fun_flat.defvjp(solver_fun_fwd, solver_fun_bwd)
        return solver_fun_flat

    def wrapped_solver_fun(*args, **kwargs):
        args, kwargs = _signature_bind(solver_fun_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())
        return make_custom_vjp_solver_fun(solver_fun, keys)(*args, *vals)

    return wrapped_solver_fun


def sparse_root_vjp(
        optimality_fun,
        support_fun,
        sol,
        args,
        cotangent,
        solve=linear_solve.solve_gmres
        # solve=linear_solve.solve_normal_cg
    ):
    # Compute the support
    support = jax.tree_util.tree_map(support_fun, sol)

    def fun_sol(sol):
        # Close over the arguments
        return optimality_fun(sol, *args)

    _, vjp_fun_sol = jax.vjp(fun_sol, sol)

    def matvec(u):
        # Mask the matvec to have a symmetric operator
        Au = vjp_fun_sol(tree_mul(u, support))[0]
        return tree_mul(Au, support)

    # Mask the cotangent depending on the support
    v = tree_scalar_mul(-1, tree_mul(cotangent, support))
    u = solve(matvec, v, maxiter=support.sum())

    def fun_args(*args):
        # Cose over the solution
        return optimality_fun(sol, *args)

    _, vjp_fun_args = jax.vjp(fun_args, *args)

    return vjp_fun_args(u)
