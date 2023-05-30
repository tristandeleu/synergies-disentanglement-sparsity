import os
import wandb
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

C_LIST = sns.color_palette("colorblind", 8)
C_LIST_DARK = sns.color_palette("dark", 8)


def configure_plt(fontsize=10, poster=True):
    """Configure matplotlib with TeX and seaborn."""
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    usetex = matplotlib.checkdep_usetex(True)
    params = {'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")


def _plot_legend_apart(ax, figname, ncol=None):
    """Plot legend apart from figure."""
    # Do all your plots with fig, ax = plt.subplots(),
    # don't call plt.legend() at the end but this instead
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
               loc="upper center")
    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig


def get_runs(filters=None):
  api = wandb.Api()
  entity, project = 'utimateteam', 'sparse-meta-learning'
  runs = api.runs(entity + "/" + project, filters=filters)

  run_list = []
  for run in runs:
      # .summary contains the output keys/values for metrics like accuracy.
      #  We call ._json_dict to omit large files
      run_dict = run.summary._json_dict

      # .config contains the hyperparameters.
      #  We remove special values that start with _.
      run_dict.update({k: v for k,v in run.config.items() if not k.startswith('_')})

      # .name is the human-readable name of the run.
      run_dict["name"] = run.name

      run_list.append(run_dict)
  return run_list
