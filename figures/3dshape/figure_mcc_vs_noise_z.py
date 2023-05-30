import pandas as pd
import matplotlib.pyplot as plt

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart, get_runs
configure_plt()

data = 'binomial_gauss'
if data == 'binomial_gauss':
  df = pd.DataFrame(
    get_runs(filters={"tags": {"$in": ["oct7_varying_z_noise_scale"]}}))
else:
  df1 = pd.DataFrame(
      get_runs(filters={"tags": {"$in": ["sep20_noisy_z"]}}))
  df2 = pd.DataFrame(
      get_runs(filters={"tags": {"$in": ["sep26_z_noise_0.25"]}}))
  df3 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep23_0.5_noisy_z_v2"]}}))
  df4 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": ["sep26_z_noise_0.75"]}}))
    # filters={"tags": {"$in": ["sep24_0.75_noisy_z"]}}))
  df5 = pd.DataFrame(get_runs(
    filters={"tags": {"$in": [
      "sep21_no_noisy_z"] } } ) )
  df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# df['z_noise_scale'][df['noisy_z'] == True] = 1



df['l1reg'] = df.solver.apply(lambda x: x.get('l1reg'))
df['l2reg'] = df.solver.apply(lambda x: x.get('l2reg'))
df['seed'] = df.data.apply(lambda x: x.get('seed'))
df['corr'] = df.z_dist.apply(lambda x: float(x[-3:]))
df['used_l1reg'] = df.l1reg.apply(lambda x: x!=0)
df['used_l2reg'] = df.l1reg.apply(lambda x: x==0)

df = df.sort_values('z_noise_scale')

groups_seed = ['seed', 'used_l1reg', 'z_noise_scale']

groups = ['used_l1reg', 'z_noise_scale']

df['max_mcc'] = df.groupby(groups_seed)['test/dis/mcc_full'].transform('max')
df['max_mcc_ica'] = df.groupby(groups_seed)['test/dis/mcc_full_ica'].transform('max')
df['max_r'] = df.groupby(groups_seed)['test/dis/r_full'].transform('max')
df['max_r_ica'] = df.groupby(groups_seed)['test/dis/r_full_ica'].transform('max')

df['avg_max_mcc'] = df.groupby(groups)['max_mcc'].transform('mean')
df['avg_max_mcc_ica'] = df.groupby(groups)['max_mcc_ica'].transform('mean')
df['avg_max_r'] = df.groupby(groups)['max_r'].transform('mean')
df['avg_max_r_ica'] = df.groupby(groups)['max_r_ica'].transform('mean')




run_l1reg = df[df['l1reg'] != 0]
run_l2reg = df[df['l1reg'] == 0]


dict_x = {}
dict_x['l1reg'] = run_l1reg['z_noise_scale']
dict_x['l2reg'] = run_l2reg['z_noise_scale']
dict_x['l2reg_ica'] = run_l2reg['z_noise_scale']

dict_y = {}
dict_y['l1reg'] = run_l1reg['avg_max_mcc']
dict_y['l2reg'] = run_l2reg['avg_max_mcc']
dict_y['l2reg_ica'] = run_l2reg['avg_max_mcc_ica']

dict_y_r = {}
dict_y_r['l1reg'] = run_l1reg['avg_max_r']
dict_y_r['l2reg'] = run_l2reg['avg_max_r']
dict_y_r['l2reg_ica'] = run_l2reg['avg_max_r_ica']

dict_label = {}
dict_label['l1reg'] = 'Lasso'
dict_label['l2reg'] = 'Ridge'
dict_label['l2reg_ica'] = 'Ridge + ICA'

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
