# setup a class to plot the utility of a given strategy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import Helpers.helpers as helpers

class UtilityPlotter:
    def __init__(self, plain_df_location, models, n_subplots = 2, sharey=False):
        self.n_subplots = n_subplots
        self.dataset = helpers.load_dataset(plain_df_location)
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

    def add_utility_plot(self, result, metric_name, epsilons,  metric = 'Adjusted Mutual Information (AMI)', title='External validation (AMI & ARI): Difference between privately trained cluster algorithms versus \n non-private trained cluster algorithms', graph_index=0):
        ax = sns.lineplot(x='epsilon', y=metric_name, data=result, ax=self.axs[graph_index], style='type', hue='type', markers=True, legend=True)
        ax.set_xticks(epsilons, labels=epsilons)
        ax.set_title(title)
        ax.set_xlabel('Privacy budget ($\epsilon$)')
        ax.set_ylabel(metric)
        ax.get_legend().remove()

    def add_baseline(self, baseline, graph_index=0):
      self.axs[graph_index].axhline(y=baseline, linestyle='--', label='non-private KMeans (baseline)')

        
    def add_legend(self, pos = None, graph_index=0): 
      handles, labels = self.axs[graph_index].get_legend_handles_labels()
      self.axs[graph_index].legend(title='Cluster algorithm', labels=labels, loc='lower center' if pos is None else pos)

    def plot_external_validation(self, utility_metrics, export_path = '../export/results/', save=True):
      self.add_utility_plot(utility_metrics, 'ari',self.plotter_data.get_epsilons(), graph_index=0)
      self.add_utility_plot(utility_metrics, 'ami',self.plotter_data.get_epsilons(), graph_index=1, title='', metric='Adjusted Rand Index (ARI)')
      #self.add_baseline(self._get_baseline(), 1)
      self.add_legend()
      if save:
         print('Save external validation plot to ' + export_path + 'ami-and-ari.png')
         self.fig.savefig(export_path + 'ami-and-ari.png')
         plt.clf()

      
    def plot_internal_validation(self, utility_metrics, export_path = '../export/results/', save=True):
       self.add_utility_plot(utility_metrics, 'ch',self.plotter_data.get_epsilons(), graph_index=0, metric='Calinski Harabasz (CH)', title='Internal validation of privately trained cluster algorithms \n using the Calinski Harabasz score and silhoutte score')
       self.add_utility_plot(utility_metrics, 'sc',self.plotter_data.get_epsilons(), graph_index=1, title='', metric='Silhouette score (SC)')
       self.add_baseline(self._get_baseline(record_type='avg_ch'), 0)
       self.add_baseline(self._get_baseline(record_type='avg_sc'), 1)
       self.add_legend()
       if save:
          self.fig.savefig(export_path + 'ch-and-sc.png')
          plt.clf()


class UtilityPlotterData: 
  def __init__(self, plain_df, n_times):
    self.plain_df = plain_df
    self.n_times = n_times

  def get_epsilons(self):
     return helpers.get_experiment_epsilons()

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

    return { 'avg_ch': np.sum(avg_ch) / self.n_times, 'avg_sc': np.sum(avg_sh) / self.n_times }