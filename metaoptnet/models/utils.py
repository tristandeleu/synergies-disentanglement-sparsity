import torch
from functorch import vmap


def proj_simplex(v):
    r"""
    Function taken from the POT library.

    Compute the closest point (orthogonal projection) on the
    generalized `(n-1)`-simplex of a vector :math:`\mathbf{v}` wrt. to the Euclidean
    distance, thus solving:

    .. math::
        \mathcal{P}(w) \in \mathop{\arg \min}_\gamma \| \gamma - \mathbf{v} \|_2

        s.t. \ \gamma^T \mathbf{1} = z

             \gamma \geq 0

    If :math:`\mathbf{v}` is a 2d array, compute all the projections wrt. axis 0

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    v : {array-like}, shape (n, d)
    z : int, optional
        'size' of the simplex (each vectors sum to z, 1 by default)

    Returns
    -------
    h : ndarray, shape (`n`, `d`)
        Array of projections on the simplex
    """
    n = v.shape[0]
    if v.ndim == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    d = v.shape[1]

    # sort u in ascending order
    # import ipdb; ipdb.set_trace()
    u, _ = torch.sort(v, dim=0)
    # u = nx.sort(v, axis=0)
    # take the descending order
    u = torch.flip(u, (0,))
    # u = nx.flip(u, 0)
    cssv = torch.cumsum(u, dim=0) - 1
    # cssv = nx.cumsum(u, dim=0) - z
    ind = torch.arange(0, n, 1, device=v.device)[:, None] + 1
    # ind = nx.arange(n, type_as=v)[:, None] + 1
    cond = u - cssv / ind > 0
    rho = torch.sum(cond, dim=0)
    theta = cssv[rho - 1, torch.arange(0, d, 1, device=v.device)] / rho
    w = torch.maximum(
        v - theta[None, :],
        torch.zeros(v.shape, dtype=v.dtype, device=v.device))
    # w = nx.maximum(
    #     v - theta[None, :],
    #     torch.zeros(v.shape, dtype=v.dtype, device=v.device))
    if d1:
        return w[:, 0]
    else:
        return w



def BST(params, lambda1):
    """Block soft-thresholding"""
    if lambda1 == 0.:
        return params
    else:
        norm_j = torch.linalg.norm(params, axis=1, keepdim=True)
        result = torch.nn.functional.relu(1 - lambda1 / norm_j) * params
        return result

# TODO add test grad inner loss
def grad_inner_loss(
        params_dual, inputs, targets_one_hot, lambda1, lambda2, dual_reg=0):
    result = inputs.mT @ (params_dual - targets_one_hot)
    result = BST(result, lambda1)
    # norm_j = torch.linalg.norm(result, axis=1, keepdim=True)
    # if lambda1 != 0:
    #     result = torch.nn.functional.relu(1 - lambda1 / norm_j) * result
    result = inputs @ result
    result += dual_reg * (params_dual - targets_one_hot)
    result /= lambda2
    result += targets_one_hot
    return result

def grad_i_inner_loss(
        params_dual, inputs, inputsT_params, inputsT_targets_one_hot, targets_one_hot,
        lambda2 : float, dual_reg : float, i: int):
    result = inputsT_params - inputsT_targets_one_hot
    result = inputs[i, :] @ result
    result += dual_reg * (params_dual[i, :] - targets_one_hot[i, :])
    result /= lambda2
    result += targets_one_hot[i, :]
    return result

def prox(params):
    return vmap(proj_simplex)(params)

def inner_step_pgd_(
        params, inputs, targets_one_hot, stepsizes, lambda1, lambda2, dual_reg=0):
    """One step on proximal gradient descent."""
    grads = grad_inner_loss(
        params, inputs, targets_one_hot, lambda1, lambda2, dual_reg)
    res = prox(params - stepsizes * grads)
    return res


@torch.jit.script
def inner_step_pcd_(
        params, inputs, inputsT_params, inputsT_targets_one_hot, targets_one_hot, stepsizes,
        lambda1 : float, lambda2 : float, dual_reg: float):
    """One epochs of proximal coordinate descent."""
    n_samples = inputs.shape[0]
    for i in range(n_samples):
        grads_i = grad_i_inner_loss(
            params, inputs, inputsT_params, inputsT_targets_one_hot,
            targets_one_hot, lambda2, dual_reg, i)
        params_i_old = params[i, :].clone()
        params[i, :] = proj_simplex(params[i, :] - stepsizes[i] * grads_i)
        inputsT_params += torch.outer(
            inputs[i, :], params[i, :] - params_i_old)
    return params, inputsT_params
