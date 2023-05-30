import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import sys

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart, get_runs
configure_plt()

# save_fig = False
save_fig = True

data = "binomial_gauss"
if data == "binomial_gauss":
    runs_df = pd.DataFrame(
        get_runs(filters={"tags": {"$in": ["oct7_support_violation"]}}))



list_task_mode = [2, 3, 6]
# list_task_mode = ["block_support_2", "block_support_3", "block_support_6"]

plt.close('all')
figsize = [14, 4]

marker = 'o'
linestyle = '--'
markersize = 8
lw = 4


fig_mcc, axarr_mcc = plt.subplots(
    1, len(list_task_mode), sharex=False, sharey=True, figsize=figsize, constrained_layout=True)
fig_r, axarr_r = plt.subplots(
    1, len(list_task_mode), sharex=False, sharey=True, figsize=figsize, constrained_layout=True)

# TODO change plam to l1reg
fontsize= 25

for idx_task_mode, block_size in enumerate(list_task_mode):

    # Sampling a sub dataframe for particular correlation value
    runs_df_subplot= runs_df[runs_df['task_mode'] == "block_support_%i" % block_size]

    # l1reg/l2reg denotes the penalty weight for Lasso/Ridge
    runs_df_subplot['l1reg'] = runs_df_subplot.solver.apply(lambda x: x.get('l1reg'))
    runs_df_subplot['l2reg'] = runs_df_subplot.solver.apply(lambda x: x.get('l2reg'))
    runs_df_subplot['seed'] = runs_df_subplot.data.apply(lambda x: x.get('seed'))

    # Differentiating between the cases l1 reg and l2 reg cases
    runs_l1_reg = runs_df_subplot[runs_df_subplot['l1reg'] != 0]
    runs_l2_reg = runs_df_subplot[runs_df_subplot['l1reg'] == 0]

    # Creating the dataframe with performance aggregated across seed values
    plot_df={}
    plot_df['penalty']= runs_l2_reg['l2reg'].unique().tolist()
    plot_df['penalty'].sort()
    #Cleaning for the correlation 0.7 case
    if 0 in plot_df['penalty']:
        plot_df['penalty'].remove(0)

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
    bar_labels = ['0.0', '0.01', '0.03', '0.1', '0.3', '1.0']

    dim = len(plot_df['penalty'])
    w = 0.75
    dimw = w / dim

    axarr_r[idx_task_mode].bar(x_ticks-0.2, plot_df['l1reg-r-mean'], yerr= plot_df['l1reg-r-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_r[idx_task_mode].bar(x_ticks, plot_df['l2reg-r-mean'], yerr= plot_df['l2reg-r-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_r[idx_task_mode].bar(x_ticks+0.2, plot_df['l2reg+ICA-r-mean'], yerr= plot_df['l2reg+ICA-r-sd'], width= dimw, label=bar_labels, capsize=5)

    axarr_mcc[idx_task_mode].bar(x_ticks-0.2, plot_df['l1reg-mcc-mean'], yerr= plot_df['l1reg-mcc-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_mcc[idx_task_mode].bar(x_ticks, plot_df['l2reg-mcc-mean'], yerr= plot_df['l2reg-mcc-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_mcc[idx_task_mode].bar(x_ticks+0.2, plot_df['l2reg+ICA-mcc-mean'], yerr= plot_df['l2reg+ICA-mcc-sd'], width= dimw, label=bar_labels, capsize=5)

    for axarr in [axarr_mcc, axarr_r]:
        axarr[idx_task_mode].set_title("Block size = %i" % block_size)
        axarr[idx_task_mode].set_xlabel('$\lambda / \lambda_{\max}$', fontsize=fontsize)
        axarr[idx_task_mode].set_xticks(x_ticks)
        axarr[idx_task_mode].set_xticklabels(bar_labels)


axarr_r[0].set_ylabel('R', fontsize=fontsize)
axarr_mcc[0].set_ylabel('MCC', fontsize=fontsize)
axarr_mcc[1].set_ylim((0.2, 1))


if save_fig:
    fig_dir = '../../../disentanglement_sparsity/figures/'
    fig_name_r = '3dshape_influ_violation_support_full_r_' + data
    fig_name_mcc = '3dshape_influ_violation_support_full_mcc_' + data

    fig_mcc.savefig(
      fig_dir + fig_name_mcc + '.pdf', bbox_inches='tight',
      dpi=600
    )
    fig_r.savefig(
      fig_dir + fig_name_r + '.pdf', bbox_inches='tight',
      dpi=600
    )

    _plot_legend_apart(axarr_mcc[1], fig_dir + '3dshape_influ_corr_full_legend.pdf')
plt.show()
