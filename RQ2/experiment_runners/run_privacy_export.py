import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from Helpers import helpers

def theoretical_limit(epsilons):
	return [np.exp(eps) - 1 for eps in epsilons]

def pretty_position(X, Y, pos):
	return ((X[pos] + X[pos+1]) / 2, (Y[pos] + Y[pos+1]) / 2)

def plot_privacy(epsilons, privacy_dataset, n_samples=2000, prefix=''): 
    avg_mi = privacy_dataset.groupby('epsilon').mean()['shokri_mi_adv']
    avg_std = privacy_dataset.groupby('epsilon').std()['shokri_mi_adv']
    avg_std = np.std(avg_std)
    fig, ax = plt.subplots(figsize=(15, 5))
    bottom, top = plt.ylim()
    #ax.errorbar(epsilons, theoretical_limit(epsilons), color='black', fmt='--', capsize=2, label='Theoretical Limit')
    ax.errorbar(epsilons, avg_mi, yerr=avg_std, fmt='.-', capsize=2)
    plt.ylim(bottom, 0.25)
    # ax.annotate("$\epsilon$-DP Bound", pretty_position(epsilons, theoretical_limit(epsilons), 4), textcoords="offset points", xytext=(5,0), ha='left')
    plt.yticks(np.arange(0, 0.26, step=0.05))
    ax.set_title('Member inference attack with dataset with shape ('+str(n_samples)+', 2)')
    ax.set_xlabel('Privacy budget ($\epsilon$)')
    plt.ylabel('Shokri adversary advantage')
    ax.set_xticks(epsilons, labels=epsilons)
    fig.savefig('./export/results/'+prefix+'shokri_privacy_adv_' + str(n_samples) +'.png')
    plt.clf()

n_samples = 2000
privacy_dataset = pd.read_csv('./export/privacy/report-advantages-'+ str(n_samples) + '.csv')
privacy_dataset_truncated = pd.read_csv('./export/privacy/truncated-report-advantages-'+ str(n_samples) + '.csv')

epsilons = helpers.get_experiment_epsilons()

plot_privacy(epsilons, privacy_dataset, n_samples=n_samples)
plot_privacy(epsilons, privacy_dataset_truncated, n_samples=n_samples, prefix='truncated-')