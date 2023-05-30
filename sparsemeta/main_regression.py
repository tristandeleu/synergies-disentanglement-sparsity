import haiku as hk
import optax

from jax import random
from conv import ConvDisentanglement
from jax_meta.utils.io import save, load
from sparse_bi_level_regression import SparseBiLevelRegression
from tqdm import tqdm

from datasets.regression_3dshapes import Regression3DShapes
from disentanglement_metrics import evaluate_disentanglement

def main(args):
    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder(inputs, is_training):
        # when using special_layer_norm, should add two dummy variables...
        outdim = args.z_dim + 2 if args.rep_norm == "special_layer_norm" else args.z_dim
        features = ConvDisentanglement(outdim)(inputs, is_training)
        if args.rep_norm == "layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=args.learn_rep_norm,
                create_offset=args.learn_rep_norm
            )(features)
        elif args.rep_norm == "special_layer_norm":
            features = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=args.learn_rep_norm,
                create_offset=args.learn_rep_norm
            )(features)[:, :-2]  # remove two dummy dimensions
        elif args.rep_norm == "batch_norm":
            features = hk.BatchNorm(
                decay_rate=0.9,
                create_scale=args.learn_rep_norm,
                create_offset=args.learn_rep_norm)(
                features, is_training=is_training)
        elif args.rep_norm is not None:
            raise NotImplementedError(f"--rep_norm {args.rep_norm} is not implemented.")
        return features

    train_dataset = eval(args.data.dataset)(
        args.data.folder,
        batch_size=args.batch_size,
        shots=args.data.shots,
        test_shots=args.data.test_shots,
        size=args.num_batches,
        split='train',
        split_mode=args.split_mode,
        seed=args.data.seed,
        download=True,
        scale_noise=args.scale_noise,
        z_dist=args.z_dist,
        z_noise_scale=args.z_noise_scale
    )

    # Evaluate (on 1000 tasks)
    val_dataset = eval(args.data.dataset)(
        args.data.folder,
        batch_size=10,
        shots=args.data.shots,
        test_shots=args.data.test_shots,
        size=100,
        split='val',
        split_mode=args.split_mode,
        task_mode=args.task_mode,
        seed=args.data.seed,
        _data=train_dataset._data,
        _factors=train_dataset._factors,
        scale_noise=args.scale_noise,
        z_dist=args.z_dist,
        z_noise_scale=args.z_noise_scale
    )

    metalearner = SparseBiLevelRegression(
        encoder,
        maxiter_inner=args.solver.maxiter_inner,
        inner_solver=args.solver.inner_solver,
        l1reg=args.solver.l1reg,
        outer_l1reg=args.outer_l1reg,
        l2reg=args.solver.l2reg,
        outer_l2reg=args.outer_l2reg,
        use_ridge_solver=args.use_ridge_solver,
        tol=args.solver.tol,
        unroll=args.unroll,
        use_plam=args.use_plam,
        inner_outer_split=(not args.no_inner_outer_split)
    )

    key = random.PRNGKey(args.data.seed)
    optimizer = optax.adamw(
        args.meta_lr,
        weight_decay=args.weight_decay
    )
    params, state = metalearner.init(key, optimizer, train_dataset.dummy_input, True)

    # Train
    sparsity_average = None
    alpha = 0.05
    best_loss, patience = None, 0
    for i, batch in enumerate(tqdm(train_dataset)):
        params, state, results = metalearner.step(
            params, state, batch['train'], batch['test'], True
        )

        # early stopping
        if (i + 1) % args.meta_val_every == 0:
            val_results = metalearner.evaluate(params, state, val_dataset, False)

            if (best_loss is None) or (val_results['outer/loss'] < best_loss):
                # Save the best model
                best_loss = val_results['outer/loss']
                patience = 0
                with open('best_model.npz', 'wb') as f:
                    save(f, params=params, state=state.model)
            else:
                patience += 1
                if (args.patience > 0) and (patience >= args.patience):
                    print(f'Early-stopping after hitting a patience of {args.patience:d} '
                          f'({i:d} meta-training iterations)')
                    break

        # exponential moving average of `sparsity`
        if sparsity_average is None:
            sparsity_average = results["outer/sparsity"].mean(0)
        else:
            sparsity_average = (1 - alpha) * sparsity_average + alpha * results["outer/sparsity"].mean(0)

    # Save the last model
    with open('latest_model.npz', 'wb') as f:
        save(f, params=params, state=state.model)

    # Load the best model
    with open('best_model.npz', 'rb') as f:
        best_model = load(f)
        params = best_model['params']
        state = state._replace(model=best_model['state'])

    # Evaluate (on 1000 tasks)
    test_dataset = eval(args.data.dataset)(
        args.data.folder,
        batch_size=10,
        shots=train_dataset.shots,
        test_shots=train_dataset.test_shots,
        size=100,
        split='test',
        split_mode=args.split_mode,
        seed=args.data.seed,
        _data=train_dataset._data,
        _factors=train_dataset._factors,
        scale_noise=args.scale_noise,
        z_dist=args.z_dist,
        z_noise_scale=args.z_noise_scale
    )

    results = metalearner.evaluate(params, state, test_dataset, False)

    # evaluate disentanglement
    dis_metrics = evaluate_disentanglement(test_dataset, encoder, params, state, sparsity_average, ica=True)
    mcc_full, r_full, mcc, r, mcc_full_ica, r_full_ica, _, _ = dis_metrics


if __name__ == '__main__':
    from simple_parsing import ArgumentParser
    from sparsemeta.utils.arguments import SolverArguments, RegressionDataArguments

    parser = ArgumentParser('Sparse Bi-Level (Regression)')

    # Data-specific arguments
    parser.add_arguments(RegressionDataArguments, dest='data')

    # Solver-specific arguments
    parser.add_arguments(SolverArguments, dest='solver')

    # Optimization
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--batch_size', type=int, default=8,
        help='number of tasks in a batch of tasks (default: %(default)s)')
    optim.add_argument('--num_batches', type=int, default=100_000,
        help='number of batch of tasks (default: %(default)s)')
    optim.add_argument('--meta_lr', type=float, default=1e-3,
        help='learning rate for meta-optimization (default: %(default)s)')
    optim.add_argument('--weight_decay', type=float, default=0.,
        help='weight decay (default: %(default)s)')


    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_filters', type=int, default=64,
        help='number of filters (default: %(default)s)')
    misc.add_argument('--unroll', action='store_true',
        help='use unrolling for meta-optimization, instead of implicit differentiation')
    misc.add_argument('--rep_norm', type=str, default=None,
                      help='Normalization strategy applied to representation')
    misc.add_argument('--conv_norm', type=str, default=None,
                      help='Normalization strategy applied inside ConvNet.')
    misc.add_argument('--learn_rep_norm', action='store_true',
        help='do not learn the normalization parameters for the representation')
    misc.add_argument('--z_dim', type=int, default=None,
        help='When --z_dim is not None, append a one hidden layer MLP (256 hidden units) '
             'to encoder with output dimension --z_dim.')
    misc.add_argument('--output_dir', type=str, default=None,
                      help='where we save files (Right now empty)')
    misc.add_argument('--use_plam', action='store_true',
                      help='Use lambda_max trick to have an adaptive regularizer.')
    misc.add_argument('--no_inner_outer_split', action='store_true',
                      help='When True, use full dataset in inner and outer problem.')
    misc.add_argument('--use_ridge_solver', action='store_true',
                      help='When --l1reg 0, will use ridge solver for inner loop.')
    misc.add_argument('--outer_l1reg', type=float, default=0.0,
                      help='Value of the l1 regularization in the outer problem')
    misc.add_argument('--outer_l2reg', type=float, default=0.0,
                      help='Value of the l2 regularization in the outer problem.')
    misc.add_argument('--encoder', type=str, default="encoder",
                      help='Encoder type.')

    misc.add_argument('--meta_val_every', type=int, default=1000,
        help='period of evaluation on meta-validation tasks (default: %(default)s)')
    misc.add_argument('--dis_eval_every', type=int, default=2500,
                      help='Number of iterations between each disentanglement evaluation (default: %(default)s)')
    misc.add_argument('--patience', type=int, default=-1,
        help='patience for early-stopping (default: %(default)s, -1: no early-stopping)')
    misc.add_argument('--split_mode', type=str, default="same",
                      help='How to split tasks into meta-train and meta-valid datasets.'
                           'same = use same dataset for both.')
    misc.add_argument('--task_mode', type=str, default="laplace",
                      help='how to sample coefficients for the tasks')
    misc.add_argument('--scale_noise', type=float, default=0.1,
        help='Noise scale (default: %(default)s)')
    misc.add_argument('--z_dist', type=str, default="uniform",
                      help='distribution over ground-truth latent factors.')
    misc.add_argument('--z_noise_scale', type=float, default=0.0,
                      help='Level of uniform noise to break the grid structure of the latents. Between 0 and 1.')

    args = parser.parse_args()

    main(args)
