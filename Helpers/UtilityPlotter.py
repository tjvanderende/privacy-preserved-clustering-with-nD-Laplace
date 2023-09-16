# setup a class to plot the utility of a given strategy
from matplotlib import pyplot as plt, ticker
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import Helpers.helpers as helpers
import pylab

class UtilityPlotter:
    def __init__(self, plain_df_location, models, n_subplots=2, sharey=False, columns=['X', 'Y']):
        self.n_subplots = n_subplots
        self.columns = columns
        self.dataset = helpers.load_dataset(plain_df_location)
        self.dataset = self.dataset[columns]
        self.models = models
        self.sharey = sharey
        self.plotter_data = UtilityPlotterData(self.dataset, 10)
        self.create_plot()

    def _get_baseline(self, record_type='ch'):
        return self.plotter_data.calculate_baseline(self.models)[record_type];

    def create_plot(self):
        fig, axs = plt.subplots(2, sharex=True, sharey=self.sharey, figsize=(12, 10))
        self.fig = fig
        self.axs = axs

    def get_color_for_cluster_algorithm(self, name: str):
        if name.__contains__('KMeans'):
            return 'blue'
        elif name.__contains__('Affinity'):
            return 'red'
        elif name.__contains__('OPTICS'):
            return 'green'
    def get_full_metric_name(self, metric_name):
        if metric_name == 'ari':
            return 'Adjusted Rand Index (ARI)'
        elif metric_name == 'ami':
            return 'Adjusted Mutual Information (AMI)'
        elif metric_name == 'ch':
            return 'Calinski Harabasz (CH)'
        elif metric_name == 'sc':
            return 'Silhouette score (SC)'

    def generate_color_palette(self, algorithms):
        return {algorithm: self.get_color_for_cluster_algorithm(algorithm) for algorithm in algorithms}

    def add_utility_plot(self, result, metric_name, epsilons, ax = None, cluster_column_name='type', metric='Adjusted Mutual Information (AMI)',
                         title='External validation (AMI & ARI): Difference between privately trained cluster algorithms versus \n non-private trained cluster algorithms',
                         graph_index=0):
        types = result[cluster_column_name].unique()
        ax = ax if ax is not None else self.axs[graph_index]
        ax = sns.lineplot(x='epsilon', y=metric_name, data=result, ax=ax, style=cluster_column_name, hue=cluster_column_name,
                          palette=self.generate_color_palette(types), markers=True, legend=True)
        ax.set_xticks(epsilons, labels=epsilons)
        ax.set_title(title)
        ax.set_xlabel('Privacy budget ($\epsilon$)')
        ax.set_ylabel(metric)
        # ax.get_legend().remove()
        return ax

    def add_baseline(self, baseline, graph_index=0, ax = None):
        ax = ax if ax is not None else self.axs[graph_index]
        ax.axhline(y=baseline, linestyle='--', label='non-private KMeans (baseline)')

    def add_legend(self, pos=None, graph_index=0, ax = None):
        ax_l = ax if ax is not None else self.axs[graph_index]
        handles, labels = ax_l.get_legend_handles_labels()
        ax_l.legend(title='Cluster algorithm', labels=labels,
                                     loc='lower center' if pos is None else pos)

    def plot_external_validation(self, utility_metrics, export_path='../export/results/', save=True):
        ax1 = self.add_utility_plot(utility_metrics, 'ari', self.plotter_data.get_epsilons(), graph_index=0)
        ax2 = self.add_utility_plot(utility_metrics, 'ami', self.plotter_data.get_epsilons(), graph_index=1, title='',
                                    metric='Adjusted Rand Index (ARI)')
        # self.add_baseline(self._get_baseline(), 1)
        # self.add_legend()
        ax1.get_legend().remove()
        ax2.legend(loc='center left')
        if save:
            print('Save external validation plot to ' + export_path + 'ami-and-ari.png')
            self.fig.savefig(export_path + 'ami-and-ari.png')
            plt.clf()

    def plot_all_metrics(self, utility_metrics, export_path='../export/results/', save=True,
                                              metrics=['ari', 'sc', 'ami', 'ch'], cluster_column_name='type', export_file_name=None):
        for i in range(len(metrics)):
            fig, ax = plt.subplots(1, sharex=True, sharey=self.sharey, figsize=(12, 10))
            ax = self.add_utility_plot(utility_metrics, metrics[i], self.plotter_data.get_epsilons(), ax=ax, graph_index=0,
                                       metric=self.get_full_metric_name(metrics[i]), title='', cluster_column_name=cluster_column_name)
            #if metrics[i] == 'ch' or metrics[i] == 'sc':
            #    self.add_baseline(self._get_baseline(record_type=f'avg_{metrics[i]}'), 0)
            self.add_legend(ax=ax)
            ax.get_legend().remove()

            if save:
                file_name = metrics[i] if export_file_name is None else export_file_name
                fig.savefig(f'{export_path}/{file_name}.png')
                plt.clf()
            else:
                plt.show()

    def plot_results_for_mechanism_comparison(self, utility_metrics, export_path='../export/results/', save=True):
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 8), linewidth=2, constrained_layout=True)
        figLegend = pylab.figure(figsize=(1.5, 1.3))
        ax_ami = self.add_utility_plot(utility_metrics, 'ami', self.plotter_data.get_epsilons(), graph_index=0,
                                    metric=self.get_full_metric_name('ami'), title='', ax=ax1)
        ax_sc = self.add_utility_plot(utility_metrics, 'sc', self.plotter_data.get_epsilons(), graph_index=1, title='', ax=ax2,
                                    metric=self.get_full_metric_name('sc'))
        self.add_baseline(self._get_baseline(record_type='avg_sc'), 1, ax=ax2)
        #self.add_legend(ax=ax2)
        ax_ami.get_legend().remove()
        ax_sc.get_legend().remove()
        ax1.grid(linestyle='dotted')
        ax2.grid(linestyle='dotted')
        ax_ami.set_title('External validation of the privacy mechanisms using AMI metric')
        ax_sc.set_title('Internal validation of the privacy mechanisms using SC metric')
        ax_ami.set_xticks(self.plotter_data.get_epsilons())

        plt.tight_layout()
        if save:
            pylab.figlegend(*ax2.get_legend_handles_labels(), loc='upper left')
            figLegend.savefig(export_path + 'legend.png', dpi=300, bbox_inches='tight')
            fig.savefig(export_path + 'ami-and-sc.png', dpi=300, bbox_inches='tight')
            plt.clf()

    def plot_internal_validation(self, utility_metrics, export_path='../export/results/', save=True):
        ax1 = self.add_utility_plot(utility_metrics, 'ch', self.plotter_data.get_epsilons(), graph_index=0,
                                    metric='Calinski Harabasz (CH)',
                                    title='Internal validation of privately trained cluster algorithms \n using the Calinski Harabasz score and silhoutte score')
        ax2 = self.add_utility_plot(utility_metrics, 'sc', self.plotter_data.get_epsilons(), graph_index=1, title='',
                                    metric='Silhouette score (SC)')
        self.add_baseline(self._get_baseline(record_type='avg_ch'), 0)
        self.add_baseline(self._get_baseline(record_type='avg_sc'), 1)
        ax1.get_legend().remove()
        ax2.legend(loc='center left')
        if save:
            self.fig.savefig(export_path + 'ch-and-sc.png')
            plt.clf()


class UtilityPlotterData:
    def __init__(self, plain_df, n_times):
        self.plain_df = plain_df
        self.n_times = n_times

    def get_epsilons(self):
        epsilons = helpers.get_experiment_epsilons()
        epsilons.remove(0.7)
        return epsilons

    def calculate_baseline(self, models):
        plain_df_scaled = StandardScaler().fit_transform(self.plain_df)
        plain_fitted_df = models['KMeans'].fit(plain_df_scaled)
        avg_ch = []
        avg_sh = []
        for i in range(self.n_times):
            ch = calinski_harabasz_score(plain_df_scaled, plain_fitted_df.labels_)
            sh = silhouette_score(plain_df_scaled, plain_fitted_df.labels_)
            avg_ch.append(ch)
            avg_sh.append(sh)

        return {'avg_ch': np.sum(avg_ch) / self.n_times, 'avg_sc': np.sum(avg_sh) / self.n_times}
