import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import sys

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart, get_runs

# save_fig = False
save_fig = True

configure_plt()

# data = 'else'

# outer = False
outer = True
# data = "else"
data = "binomial_gauss"
if outer is True:
    df = pd.DataFrame(
        get_runs(filters={"tags": {"$in": ["oct19_outer_reg"]}}))
elif data == 'binomial_gauss':
  df = pd.DataFrame(
    get_runs(filters={"tags": {"$in": ["oct7_varying_z_noise_scale"]}}))
else:
    df1 = pd.DataFrame(
        get_runs(filters={"tags": {"$in": ["sep20_noisy_z"]}}))
    df2 = pd.DataFrame(
        get_runs(filters={"tags": {"$in": ["sep26_z_noise_0.25"]}}))
        # get_runs(filters={"tags": {"$in": ["sep23_0.25_noisy_z_v2"]}}))
    df3 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep23_0.5_noisy_z_v2"]}}))
    df4 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep26_z_noise_0.75_v2"]}}))
    df5 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep21_no_noisy_z"] } } ) )
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)


list_noise_scale = [1, 1]
# list_noise_scale = [0, 1]
plt.close('all')
figsize = [14, 2.75]


marker = 'o'
linestyle = '--'
markersize = 8
lw = 4

if outer:
    df['l1reg'] = df.outer_l1reg
    df['l2reg'] = df.outer_l2reg
else:
    df['l1reg'] = df.solver.apply(lambda x: x.get('l1reg'))
    df['l2reg'] = df.solver.apply(lambda x: x.get('l2reg'))
    df['seed'] = df.data.apply(lambda x: x.get('seed'))

fig_mcc, axarr_mcc = plt.subplots(
    1, 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)

fig_r, axarr_r = plt.subplots(
    1, 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)

# TODO change plam to l1reg
fontsize= 25

# for idx_cor, cor in enumerate(list_noise_scale):
cor = 1
# Sampling a sub dataframe for particular correlation value
# df_subplot= df[df['z_dist'] == 'harder_gauss_%s' % cor]
df_subplot = df[df['z_noise_scale'] == cor]

# l1reg/l2reg denotes the penalty weight for Lasso/Ridge


# Differentiating between the cases l1 reg and l2 reg cases
runs_l1_reg = df_subplot[df_subplot['l1reg'] != 0]
runs_l2_reg = df_subplot[df_subplot['l1reg'] == 0]

# Creating the dataframe with performance aggregated across seed values
plot_df={}
plot_df['penalty']= runs_l2_reg['l2reg'].unique().tolist()
plot_df['penalty'].sort()

for case in ['l1reg', 'l2reg', 'l2reg+ICA']:
    plot_df[case + '-mcc-mean']= []
    plot_df[case + '-mcc-sd']= []
    plot_df[case + '-r-mean']= []
    plot_df[case + '-r-sd']= []

    for penalty in plot_df['penalty']:

        if case == 'l1reg':
            indices= runs_l1_reg['l1reg'] == penalty
            arr_mcc= runs_l1_reg[indices]['test/dis/mcc_full']
            arr_r = runs_l1_reg[indices]['test/dis/r_full']
        elif case == 'l2reg':
            indices= runs_l2_reg['l2reg'] == penalty
            arr_mcc= runs_l2_reg[indices]['test/dis/mcc_full']
            arr_r = runs_l2_reg[indices]['test/dis/r_full']
        elif case == 'l2reg+ICA':
            indices= runs_l2_reg['l2reg'] == penalty
            arr_mcc= runs_l2_reg[indices]['test/dis/mcc_full_ica']
            arr_r = runs_l2_reg[indices]['test/dis/r_full_ica']

        plot_df[case + '-mcc-mean'].append(arr_mcc.mean() )
        plot_df[case + '-mcc-sd'].append(arr_mcc.std() )
        plot_df[case + '-r-mean'].append(arr_r.mean() )
        plot_df[case + '-r-sd'].append(arr_r.std() )

#Bar Plot
x_ticks = np.arange(len(plot_df['penalty']))
if outer:
    bar_labels = df.l1reg.unique()
    bar_labels.sort()
else:
    bar_labels = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]

dim = len(plot_df['penalty'])
if not outer:
    w = 0.75
else:
    w = 2
dimw = w / dim

idx_cor = 0

axarr_r.bar(x_ticks-0.2, plot_df['l1reg-r-mean'], yerr= plot_df['l1reg-r-sd'], width= dimw, label=bar_labels, capsize=5)
axarr_r.bar(x_ticks, plot_df['l2reg-r-mean'], yerr= plot_df['l2reg-r-sd'], width= dimw,  label=bar_labels, capsize=5)
axarr_r.bar(x_ticks+0.2, plot_df['l2reg+ICA-r-mean'], yerr= plot_df['l2reg+ICA-r-sd'], width= dimw, label=bar_labels, capsize=5)

# elif int(sys.argv[1]) == 1:
axarr_mcc.bar(x_ticks-0.2, plot_df['l1reg-mcc-mean'], yerr= plot_df['l1reg-mcc-sd'], width= dimw, label=bar_labels, capsize=5)
axarr_mcc.bar(x_ticks, plot_df['l2reg-mcc-mean'], yerr= plot_df['l2reg-mcc-sd'], width= dimw,  label=bar_labels, capsize=5)
axarr_mcc.bar(x_ticks+0.2, plot_df['l2reg+ICA-mcc-mean'], yerr= plot_df['l2reg+ICA-mcc-sd'], width= dimw, label=bar_labels, capsize=5)

for axarr in [axarr_mcc, axarr_r]:
    axarr.set_title("Noise scale = %s" % cor)
    axarr.set_xlabel('$\lambda / \lambda_{\max}$', fontsize=fontsize)
    axarr.set_xticks(x_ticks)
    axarr.set_xticklabels(bar_labels)

axarr_mcc.set_ylabel('MCC', fontsize=fontsize)
axarr_r.set_ylabel('R', fontsize=fontsize)
if not outer:
    axarr_r[0].set_ylabel('R', fontsize=fontsize)
    axarr_r[0].set_yticks((0.4, 0.6, 0.8, 1))
    axarr_r[0].set_yticklabels((0.4, 0.6, 0.8, 1))
    axarr_r[0].set_ylim((0.4, 1))

    axarr_mcc[1].set_ylim((0.32, 1))


if save_fig:
    fig_dir = '../../../disentanglement_sparsity/figures/'
    fig_name_r = 'outer_reg_r_' + data
    fig_name_mcc = 'outer_reg_mcc_' + data
    fig_r.savefig(
      fig_dir + fig_name_r + '.pdf', bbox_inches='tight', dpi=600)
    fig_mcc.savefig(
      fig_dir + fig_name_mcc + '.pdf', bbox_inches='tight', dpi=600)
plt.show()
