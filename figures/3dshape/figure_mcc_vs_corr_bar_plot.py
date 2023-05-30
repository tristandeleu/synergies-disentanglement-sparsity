import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart, get_runs
configure_plt()

from figure_mcc_vs_corr import dict_x, dict_y, dict_label, dict_y_r, dict_y_dci_d, dict_y_dci_c

# save_fig = False
save_fig = True



# data = "else"
data = "binomial_gauss"
if data == "binomial_gauss":
    runs_df = pd.DataFrame(
        get_runs(filters={"tags": {"$in": ["oct7_varying_corr"]}}))
else:
    runs_df_cor = pd.DataFrame(get_runs(filters={"tags": {"$in": ["sep20_noisy_z"]}}))

    runs_df_less_cor = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep20_noisy_z_less_corr"]}}))
    #Removing logs for correlation 0.5 to be replaced with new results for that case
    runs_df_less_cor.drop(runs_df_less_cor[runs_df_less_cor['z_dist']=='harder_gauss_0.5'].index, inplace=True)

    runs_df_corr_zero= pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep26_corr_0"]}}))

    runs_df_corr_half= pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep26_corr_0.5"]}}))

    run_df_corr_new = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep27_z_noise_1_corr_095"]}}))

    runs_df = pd.concat(
        [runs_df_corr_zero, runs_df_corr_half, runs_df_cor, runs_df_less_cor, run_df_corr_new], ignore_index=True)


list_cor = ['0.0', '0.9']
plt.close('all')
figsize = [14, 4]



marker = 'o'
linestyle = '--'
markersize = 8
lw = 4


fig_mcc, axarr_mcc = plt.subplots(
    1, len(list_cor)+ 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)
fig_r, axarr_r = plt.subplots(
    1, len(list_cor)+ 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)
fig_dci_d, axarr_dci_d = plt.subplots(
    1, len(list_cor)+ 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)
fig_dci_c, axarr_dci_c = plt.subplots(
    1, len(list_cor)+ 1, sharex=False, sharey=True, figsize=figsize, constrained_layout=True)

# TODO change plam to l1reg
fontsize= 25

for idx_cor, cor in enumerate(list_cor):

    # Sampling a sub dataframe for particular correlation value
    runs_df_subplot= runs_df[runs_df['z_dist'] == 'harder_gauss_%s' % cor]

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
        
        #DCI Metrics
        plot_df[case + '-dci-d-mean']= []
        plot_df[case + '-dci-d-sd']= []
        plot_df[case + '-dci-c-mean']= []
        plot_df[case + '-dci-c-sd']= []
        
        for penalty in plot_df['penalty']:
            

            if case == 'l1reg':
                indices= runs_l1_reg['l1reg'] == penalty
                
                arr_mcc= runs_l1_reg[indices]['test/dis/mcc_full']
                arr_r = runs_l1_reg[indices]['test/dis/r_full']
                
                arr_dci_d = runs_l1_reg[indices]['dci_d']
                arr_dci_c = runs_l1_reg[indices]['dci_c']
                
            elif case == 'l2reg':
                indices= runs_l2_reg['l2reg'] == penalty
                
                arr_mcc= runs_l2_reg[indices]['test/dis/mcc_full']
                arr_r = runs_l2_reg[indices]['test/dis/r_full']
                
                arr_dci_d = runs_l2_reg[indices]['dci_d']
                arr_dci_c = runs_l2_reg[indices]['dci_c']
                
                
            elif case == 'l2reg+ICA':                
                indices= runs_l2_reg['l2reg'] == penalty
                
                arr_mcc= runs_l2_reg[indices]['test/dis/mcc_full_ica']
                arr_r = runs_l2_reg[indices]['test/dis/r_full_ica']
                
                arr_dci_d = runs_l2_reg[indices]['dci_d_ica']
                arr_dci_c = runs_l2_reg[indices]['dci_c_ica']
                                

            plot_df[case + '-mcc-mean'].append(arr_mcc.mean() )
            plot_df[case + '-mcc-sd'].append(arr_mcc.std() )
            plot_df[case + '-r-mean'].append(arr_r.mean() )
            plot_df[case + '-r-sd'].append(arr_r.std() )
            
            plot_df[case + '-dci-d-mean'].append(arr_dci_d.mean())
            plot_df[case + '-dci-d-sd'].append(arr_dci_d.std())
            plot_df[case + '-dci-c-mean'].append(arr_dci_c.mean())
            plot_df[case + '-dci-c-sd'].append(arr_dci_c.std())
            

    #Bar Plot
    x_ticks = np.arange(len(plot_df['penalty']))
    bar_labels = ['0.0', '0.01', '0.03', '0.1', '0.3', '1.0']

    dim = len(plot_df['penalty'])
    w = 0.75
    dimw = w / dim

    axarr_r[idx_cor].bar(x_ticks-0.2, plot_df['l1reg-r-mean'], yerr= plot_df['l1reg-r-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_r[idx_cor].bar(x_ticks, plot_df['l2reg-r-mean'], yerr= plot_df['l2reg-r-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_r[idx_cor].bar(x_ticks+0.2, plot_df['l2reg+ICA-r-mean'], yerr= plot_df['l2reg+ICA-r-sd'], width= dimw, label=bar_labels, capsize=5)

    axarr_mcc[idx_cor].bar(x_ticks-0.2, plot_df['l1reg-mcc-mean'], yerr= plot_df['l1reg-mcc-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_mcc[idx_cor].bar(x_ticks, plot_df['l2reg-mcc-mean'], yerr= plot_df['l2reg-mcc-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_mcc[idx_cor].bar(x_ticks+0.2, plot_df['l2reg+ICA-mcc-mean'], yerr= plot_df['l2reg+ICA-mcc-sd'], width= dimw, label=bar_labels, capsize=5)

    axarr_dci_d[idx_cor].bar(x_ticks-0.2, plot_df['l1reg-dci-d-mean'], yerr= plot_df['l1reg-dci-d-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_dci_d[idx_cor].bar(x_ticks, plot_df['l2reg-dci-d-mean'], yerr= plot_df['l2reg-dci-d-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_dci_d[idx_cor].bar(x_ticks+0.2, plot_df['l2reg+ICA-dci-d-mean'], yerr= plot_df['l2reg+ICA-dci-d-sd'], width= dimw, label=bar_labels, capsize=5)

    axarr_dci_c[idx_cor].bar(x_ticks-0.2, plot_df['l1reg-dci-c-mean'], yerr= plot_df['l1reg-dci-c-sd'], width= dimw, label=bar_labels, capsize=5)
    axarr_dci_c[idx_cor].bar(x_ticks, plot_df['l2reg-dci-c-mean'], yerr= plot_df['l2reg-dci-c-sd'], width= dimw,  label=bar_labels, capsize=5)
    axarr_dci_c[idx_cor].bar(x_ticks+0.2, plot_df['l2reg+ICA-dci-c-mean'], yerr= plot_df['l2reg+ICA-dci-c-sd'], width= dimw, label=bar_labels, capsize=5)
    
    for axarr in [axarr_mcc, axarr_r, axarr_dci_d, axarr_dci_c]:
        axarr[idx_cor].set_title("Correlation = %s" % cor)
        axarr[idx_cor].set_xlabel('$\lambda / \lambda_{\max}$', fontsize=fontsize)
        axarr[idx_cor].set_xticks(x_ticks)
        axarr[idx_cor].set_xticklabels(bar_labels)


for method in ['l1reg', 'l2reg', 'l2reg_ica']:
    axarr_r[len(list_cor)].plot(
        dict_x[method], dict_y_r[method],
        marker=marker, label=dict_label[method], linestyle=linestyle,
        markersize=markersize, lw=lw)
    
    axarr_mcc[len(list_cor)].plot(
        dict_x[method], dict_y[method],
        marker=marker, label=dict_label[method], linestyle=linestyle,
        markersize=markersize, lw=lw)

    axarr_dci_d[len(list_cor)].plot(
        dict_x[method], dict_y_dci_d[method],
        marker=marker, label=dict_label[method], linestyle=linestyle,
        markersize=markersize, lw=lw)

    axarr_dci_c[len(list_cor)].plot(
        dict_x[method], dict_y_dci_c[method],
        marker=marker, label=dict_label[method], linestyle=linestyle,
        markersize=markersize, lw=lw)
    
axarr_mcc[2].set_xlim((0.5, 1))

axarr_r[0].set_ylabel('R', fontsize=fontsize)
axarr_mcc[0].set_ylabel('MCC', fontsize=fontsize)
axarr_dci_d[0].set_ylabel('DCI-Disentanglement', fontsize=fontsize)
axarr_dci_c[0].set_ylabel('DCI-Completeness', fontsize=fontsize)

axarr_mcc[1].set_ylim((0.35, 1))

for axarr in [axarr_mcc, axarr_r, axarr_dci_d, axarr_dci_c]:
    axarr[len(list_cor)].set_xlabel('Correlation')
    axarr[len(list_cor)].set_title('Multiple correlation values')

if save_fig:
    fig_dir = '../../../disentanglement_sparsity/figures/'
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # if int(sys.argv[1]) == 0:
    fig_name_r = '3dshape_influ_corr_full_r_' + data
    # elif int(sys.argv[1]) == 1:
    fig_name_mcc = '3dshape_influ_corr_full_mcc_' + data
    
    fig_name_dci_d = '3dshape_influ_corr_full_dci_d_' + data
    fig_name_dci_c = '3dshape_influ_corr_full_dci_c_' + data

    fig_mcc.savefig(
      fig_dir + fig_name_mcc + '.pdf', bbox_inches='tight',
      dpi=600
    )
    fig_r.savefig(
      fig_dir + fig_name_r + '.pdf', bbox_inches='tight',
      dpi=600
    )
    fig_dci_d.savefig(
      fig_dir + fig_name_dci_d + '.pdf', bbox_inches='tight',
      dpi=600
    )
    fig_dci_c.savefig(
      fig_dir + fig_name_dci_c + '.pdf', bbox_inches='tight',
      dpi=600
    )

#     _plot_legend_apart(axarr_mcc[1], fig_dir + '3dshape_influ_corr_fulllegend.pdf')
plt.show()
