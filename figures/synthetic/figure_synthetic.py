import matplotlib
import matplotlib.pyplot as plt
import pickle

from sparsemeta.plot_utils import configure_plt, _plot_legend_apart

save_fig = True

configure_plt()

res= pickle.load(open('logs.p', 'rb'))

fontsize= 25
case='lasso'
num_samples_list= [25, 50, 75, 100, 125, 150]
marker_list = {}
marker_list['lasso'] = 'o'
marker_list['ridge'] = '^'
dict_label = {}
dict_label[0, 'lasso'] = 'Entangled-Lasso'
dict_label[0, 'ridge'] = 'Entangled-Ridge'
dict_label[1, 'lasso'] = 'Disentangled-Lasso'
dict_label[1, 'ridge'] = 'Disentangled-Ridge'

dict_start_label = {}
dict_start_label['ent-'] = 'Entangled'
dict_start_label['disent-'] = 'Disentangled'

dict_end_label = {}
dict_end_label['lasso'] = 'Lasso'
dict_end_label['ridge'] = 'Ridge'
# matplotlib.rcParams.update({'errorbar.capsize': 2})

fig, axarr = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)

dict_color = {}
dict_color['ent-'] = 'C0'
dict_color['disent-'] = 'C1'

dict_ls = {}
dict_ls['ent-'] = '-'
dict_ls['disent-'] = '--'

dict_alpha = {}
dict_alpha['ent-'] = 0.9
dict_alpha['disent-'] = 0.9

for idx_arr, n_features in enumerate([5, 20, 80]):
    # case='lasso'
    axarr[idx_arr].tick_params(labelsize=fontsize)
    # axarr[idx_arr].set_xticklabels(num_samples_list, rotation=25)
    axarr[idx_arr].set_xlabel('\# samples', fontsize=fontsize)
    axarr[idx_arr].set_title(
        ('$ \ell / m = %i$' % n_features) + r'\%', fontsize=fontsize)
    for est in ['lasso', 'ridge']:
        for ent_state in ['ent-', 'disent-']:
            axarr[idx_arr].errorbar(
                num_samples_list, res[n_features / 100][ent_state + est]['mean'],
                yerr=res[0.05][ent_state + est]['s.e.'],
                marker= marker_list[est],
                color=dict_color[ent_state], ls=dict_ls[ent_state],
                lw=5,  mew=2.5, ms=15, alpha=0.8, capsize=6,
                label='%s-%s' % (
                    dict_start_label[ent_state],
                    dict_end_label[est]))
axarr[0].set_ylabel('$R^2$', fontsize=fontsize)
axarr[0].set_ylim(-0.25, 1.0)
axarr[0].set_yticks((0, 0.5, 1))



lines, labels = axarr[0].get_legend_handles_labels()
lgd = fig.legend(
    lines, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.85), fontsize=fontsize, ncol=2)
plt.tight_layout()

if save_fig:
    fig_dir = '../../../disentanglement_sparsity/figures/'
    fig_name = 'sparsity-disentanglement-gains'

    # _plot_legend_apart(axarr[1], fig_dir + fig_name + 'legend.pdf')
    plt.savefig(
        fig_dir + fig_name + '.pdf', bbox_inches='tight',
        bbox_extra_artists=(lgd,), dpi=600)

plt.show(block=False)
