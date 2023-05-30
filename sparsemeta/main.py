import haiku as hk
import optax

from jax import random
from jax_meta.modules.conv import Conv4
from jax_meta.utils.io import save, load
from sparse_bi_level import SparseBiLevel, SparseBiLevelState


def main(args):
    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder(inputs, is_training):
        features = Conv4(num_filters=args.num_filters)(inputs, is_training)
        if args.norm:
            features = hk.LayerNorm(
                axis=1,
                create_scale=args.learn_layer_norm,
                create_offset=args.learn_layer_norm
            )(features)
        return features

    train_dataset = args.data.dataset._cls(
        args.data.folder,
        batch_size=args.batch_size,
        shots=args.data.shots,
        test_shots=args.data.test_shots,
        ways=args.data.ways,
        size=args.num_batches,
        split='train',
        data_augmentation=False,
        seed=args.data.seed,
        download=True
    )

    val_dataset = args.data.dataset._cls(
        args.data.folder,
        batch_size=10,
        shots=train_dataset.shots,
        test_shots=train_dataset.test_shots,
        ways=train_dataset.ways,
        size=100,
        split='val',
        seed=args.data.seed
    )

    metalearner = SparseBiLevel(
        encoder,
        num_ways=train_dataset.ways,
        inner_solver=args.solver.inner_solver,
        maxiter_inner=args.solver.maxiter_inner,
        est=args.solver.est,
        l1reg=args.solver.l1reg,
        l2reg=args.solver.l2reg,
        tol=args.solver.tol,
        unroll=args.unroll,
    )

    key = random.PRNGKey(args.data.seed)
    optimizer = optax.multi_transform({
        'model': optax.adamw(args.meta_lr, weight_decay=args.weight_decay),
        'log_scale': optax.adamw(
            args.meta_lr if args.enable_scale else 0.,
            weight_decay=args.weight_decay
        ),
        'log_plam': optax.adamw(
            args.meta_lr if args.learn_plam else 0.,
            weight_decay=args.weight_decay
        )
    }, SparseBiLevelState(model='model', log_scale='log_scale', log_plam='log_plam'))
    params, state = metalearner.init(key, optimizer, train_dataset.dummy_input, True)

    # Train
    best_accuracy, patience = None, 0
    for i, batch in enumerate(train_dataset):
        # Evaluation on meta-validation tasks
        if (i + 1) % args.meta_val_every == 0:
            results = metalearner.evaluate(params, state, val_dataset, False)

            if (best_accuracy is None) or (results['outer/accuracy'] > best_accuracy):
                # Save the best model
                best_accuracy = results['outer/accuracy']
                patience = 0
                with open('best_model.npz', 'wb') as f:
                    save(f,
                        params_model=params.model,
                        params_log_scale=params.log_scale,
                        state=state.model
                    )
            else:
                patience += 1
                if (args.patience > 0) and (patience >= args.patience):
                    print(f'Early-stopping after hitting a patience of {args.patience:d} '
                          f'({i:d} meta-training iterations)')
                    break

        # Meta-training
        params, state, results = metalearner.step(
            params, state, batch['train'], batch['test'], True
        )

    # Save the last model
    with open('last_model.npz', 'wb') as f:
        save(f,
            params_model=params.model,
            params_log_scale=params.log_scale,
            params_log_plam=params.log_plam,
            state=state.model,
        )

    # Load the best model
    with open('best_model.npz', 'rb') as f:
        best_model = load(f)
        params = SparseBiLevelState(
            model=best_model['params_model'],
            log_scale=best_model['params_log_scale'],
            log_plam=best_model['params_log_plam'],
        )
        state = state._replace(model=best_model['state'])

    # Evaluate (on 1000 meta-test tasks)
    test_dataset = args.data.dataset._cls(
        args.data.folder,
        batch_size=10,
        shots=train_dataset.shots,
        test_shots=train_dataset.test_shots,
        ways=train_dataset.ways,
        size=100,
        split='test',
        seed=args.data.seed
    )
    results = metalearner.evaluate(params, state, test_dataset, False)


if __name__ == '__main__':
    from simple_parsing import ArgumentParser
    from jax_meta.utils.arguments import DataArguments
    from sparsemeta.utils.arguments import SolverArguments

    parser = ArgumentParser('Sparse Bi-Level')

    # Data-specific arguments
    parser.add_arguments(DataArguments, dest='data')

    # Solver-specific arguments
    parser.add_arguments(SolverArguments, dest='solver')

    # Optimization
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--batch_size', type=int, default=8,
        help='number of tasks in a batch of tasks (default: %(default)s)')
    optim.add_argument('--num_batches', type=int, default=50_000,
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
    misc.add_argument('--no_norm', action='store_true',
        help='no norm on encoder output')
    misc.add_argument('--no_learn_layer_norm', action='store_true',
        help='do not learn the layer norm parameters')
    misc.add_argument('--enable_scale', action='store_true',
        help='scaling parameter to learn before logit')
    misc.add_argument('--learn_plam', action='store_true',
        help='meta-learn plam (with initial value --plam)')
    misc.add_argument('--meta_val_every', type=int, default=1000,
        help='period of evaluation on meta-validation tasks (default: %(default)s)')
    misc.add_argument('--patience', type=int, default=-1,
        help='patience for early-stopping (default: %(default)s, -1: no early-stopping)')

    args = parser.parse_args()

    args.learn_layer_norm = (not args.no_learn_layer_norm)
    del args.no_learn_layer_norm
    args.norm = (not args.no_norm)
    del args.no_norm

    main(args)
