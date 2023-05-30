import pandas as pd
import matplotlib.pyplot as plt

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart, get_runs
configure_plt()



# df1 = pd.DataFrame(
#     get_runs(filters={"tags": {"$in": ["sep20_noisy_z"]}}))
# df2 = pd.DataFrame(get_runs(
#   filters={"tags": {"$in": ["sep20_noisy_z_less_corr"]}}))
# df3 = pd.DataFrame(get_runs(
#   filters={"tags": {"$in": ["sep27_z_noise_1_corr_095"]}}))
# runs_df = pd.concat([df1, df2, df3], ignore_index=True)
runs_df = pd.DataFrame(
    get_runs(filters={"tags": {"$in": ["oct7_varying_corr"]}}))


runs_df['l1reg'] = runs_df.solver.apply(lambda x: x.get('l1reg'))
runs_df['l2reg'] = runs_df.solver.apply(lambda x: x.get('l2reg'))
runs_df['seed'] = runs_df.data.apply(lambda x: x.get('seed'))
runs_df['corr'] = runs_df.z_dist.apply(lambda x: float(x[-3:]))
runs_df['used_l1reg'] = runs_df.l1reg.apply(lambda x: x!=0)
runs_df['used_l2reg'] = runs_df.l1reg.apply(lambda x: x==0)

runs_df = runs_df.sort_values('corr')

groups_seed = ['seed', 'used_l1reg', 'corr']
groups = ['used_l1reg', 'corr']
runs_df['max_mcc'] = runs_df.groupby(groups_seed)['test/dis/mcc_full'].transform('max')
runs_df['max_mcc_ica'] = runs_df.groupby(groups_seed)['test/dis/mcc_full_ica'].transform('max')
runs_df['max_r'] = runs_df.groupby(groups_seed)['test/dis/r_full'].transform('max')
runs_df['max_r_ica'] = runs_df.groupby(groups_seed)['test/dis/r_full_ica'].transform('max')

runs_df['max_dci_d'] = runs_df.groupby(groups_seed)['dci_d'].transform('max')
runs_df['max_dci_d_ica'] = runs_df.groupby(groups_seed)['dci_d_ica'].transform('max')

runs_df['max_dci_c'] = runs_df.groupby(groups_seed)['dci_c'].transform('max')
runs_df['max_dci_c_ica'] = runs_df.groupby(groups_seed)['dci_c_ica'].transform('max')

runs_df['avg_max_mcc'] = runs_df.groupby(groups)['max_mcc'].transform('mean')
runs_df['avg_max_mcc_ica'] = runs_df.groupby(groups)['max_mcc_ica'].transform('mean')
runs_df['avg_max_r'] = runs_df.groupby(groups)['max_r'].transform('mean')
runs_df['avg_max_r_ica'] = runs_df.groupby(groups)['max_r_ica'].transform('mean')

runs_df['avg_max_dci_d'] = runs_df.groupby(groups)['max_dci_d'].transform('mean')
runs_df['avg_max_dci_d_ica'] = runs_df.groupby(groups)['max_dci_d_ica'].transform('mean')

runs_df['avg_max_dci_c'] = runs_df.groupby(groups)['max_dci_c'].transform('mean')
runs_df['avg_max_dci_c_ica'] = runs_df.groupby(groups)['max_dci_c_ica'].transform('mean')


list_cor = ['0.5', '0.8', '0.9', '0.95', '0.99']
figsize = [4, 3.5]

run_l1reg = runs_df[runs_df['l1reg'] != 0]
run_l2reg = runs_df[runs_df['l1reg'] == 0]


dict_x = {}
dict_x['l1reg'] = run_l1reg['corr']
dict_x['l2reg'] = run_l2reg['corr']
dict_x['l2reg_ica'] = run_l2reg['corr']

dict_y = {}
dict_y['l1reg'] = run_l1reg['avg_max_mcc']
dict_y['l2reg'] = run_l2reg['avg_max_mcc']
dict_y['l2reg_ica'] = run_l2reg['avg_max_mcc_ica']

dict_y_r = {}
dict_y_r['l1reg'] = run_l1reg['avg_max_r']
dict_y_r['l2reg'] = run_l2reg['avg_max_r']
dict_y_r['l2reg_ica'] = run_l2reg['avg_max_r_ica']

dict_y_dci_d = {}
dict_y_dci_d['l1reg'] = run_l1reg['avg_max_dci_d']
dict_y_dci_d['l2reg'] = run_l2reg['avg_max_dci_d']
dict_y_dci_d['l2reg_ica'] = run_l2reg['avg_max_dci_d_ica']

dict_y_dci_c = {}
dict_y_dci_c['l1reg'] = run_l1reg['avg_max_dci_c']
dict_y_dci_c['l2reg'] = run_l2reg['avg_max_dci_c']
dict_y_dci_c['l2reg_ica'] = run_l2reg['avg_max_dci_c_ica']


dict_label = {}
dict_label['l1reg'] = 'inner-Lasso'
dict_label['l2reg'] = 'inner-Ridge'
dict_label['l2reg_ica'] = 'inner-Ridge + ICA'

marker = '^'
linestyle = '--'
markersize = 8
lw = 4

# plt.close('all')
# fig, axarr = plt.subplots(
#     2, 1, sharey=False, sharex=True, figsize=figsize, constrained_layout=True)



# axarr[0].set_ylabel('R')
# axarr[0].set_yticks((0.9, 1))
# axarr[1].set_ylabel('MCC')
# axarr[1].set_yticks((0.7, 1))
# axarr[1].set_xlabel('Correlation')
# plt.legend()
# plt.show()
