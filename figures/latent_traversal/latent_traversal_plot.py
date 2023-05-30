import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from glob import glob
from sparsemeta.plot_utils import configure_plt


def main(args):
    configure_plt()

    filenames = glob(args.input_filename)

    for filename in filenames:
        with open(filename, 'rb') as f:
            data = np.load(f)

            factor_name = data['factor']
            run_id = data['run_id']
            images = data['images']
            representations = data['representations']

        num_images = np.minimum(len(images), args.num_images)
        indices = np.round(np.linspace(0, len(images) - 1, num_images)).astype(int)

        fig = plt.figure(figsize=(20, 6), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=num_images, nrows=3, figure=fig)

        for i, idx in enumerate(indices):
            ax = fig.add_subplot(spec[-1, i])
            ax.imshow(images[idx])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect('equal', 'box')

        ax = fig.add_subplot(spec[:-1, :])
        for j in range(representations.shape[1]):
            ax.plot(representations[:, j], label=f'$z_{{{j+1}}}$', lw=4, c=f'C{j}')
        ax.xaxis.set_visible(False)
        ax.margins(x=0)
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=(1.1, 0.5))
        ax.grid(ls=(0, (4, 3)))

        if args.output_folder is not None:
            filename = args.output_folder / f'response_{factor_name}_{run_id}.pdf'
            print(filename)
            plt.savefig(filename, bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description='Plot latent traversal')
    parser.add_argument('input_filename', type=str,
        help='Path to the filename saved with `get_representations.py`')
    parser.add_argument('--num_images', type=int, default=9,
        help='Number of images to display')
    parser.add_argument('--output_folder', type=Path, default=None,
        help='Output folder for the plot.')

    args = parser.parse_args()

    main(args)
