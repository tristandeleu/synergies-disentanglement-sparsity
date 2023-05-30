import os

from typing import Optional

from dataclasses import dataclass
from simple_parsing import choice
from simple_parsing.helpers import Serializable


@dataclass
class SolverArguments(Serializable):
    # name of the estimator (default: %(default)s)
    inner_solver: str = choice(['pgd', 'pcd', 'qp', 'qp_primal'], default='pcd')

    # maximum number of steps (default: %(default)s)
    maxiter_inner: int = 50

    # value of the regularization parameter
    # for the Lasso base learner (normalized, default: %(default)s)
    l1reg: float = 5e-2

    # value of the regularization parameter
    # for the SVM base learner
    l2reg: float = 1.0

    # tolerance for convergence (default: %(default)s)
    tol: float = 1e-3

    # name of the estimator (default: %(default)s)
    # est: str = choice(['sparse_logreg', 'svm'], default='sparse_logreg')
    est: str = choice(
        ['sparse_logreg', 'sparse_logreg_intercept','svm', "lasso",
        "group_enet_svm"],
        default='sparse_logreg')


@dataclass
class RegressionDataArguments(Serializable):
    # data folder
    folder: Optional[str]

    # dataset name
    dataset: str = choice(['Regression3DShapes'], default='Regression3DShapes')

    # number of training examples per class (k in "k-shot", default: %(default)s)
    shots: int = 5

    # number of test examples per class
    test_shots: int = 15

    # random seed
    seed: int = 1

    def __post_init__(self):
        if self.folder is None:
            if os.getenv('SLURM_TMPDIR') is not None:
                self.folder = os.path.join(os.getenv('SLURM_TMPDIR'), 'data')
            else:
                raise ValueError(f'Invalid value of `folder`: {self.folder}. '
                    '`folder` must be a valid folder.')
        os.makedirs(self.folder, exist_ok=True)

        if self.test_shots <= 0:
            self.test_shots = self.shots
