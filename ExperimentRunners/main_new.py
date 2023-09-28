import os
import sys

import numpy as np
import pandas as pd
import pylab
import typer
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, ShadowModels
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from art.utils import to_categorical
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import clone
from sklearn.cluster import KMeans, AffinityPropagation, OPTICS, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, silhouette_score, \
    roc_curve, precision_score, recall_score, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from Helpers import helpers, rq3_helpers
from Helpers.UtilityPlotter import UtilityPlotter

font_sizes = {
    'normal': 16,
    'title': 20,
}
mechanisms = ['nd-Laplace', 'piecewise']
mechanism_mapper = {
    'nd-Laplace': {
        2: helpers.get_mechanism('2d-laplace'),
        3: helpers.get_mechanism('3d-laplace'),
        'nd': helpers.get_mechanism('nd-laplace'),
    },
    'grid-nd-Laplace': {
        2: helpers.get_mechanism('2d-laplace-truncated'),
        3: helpers.get_mechanism('3d-laplace-truncated'),
        'nd': helpers.get_mechanism('nd-laplace-truncated'),
    },
    'density-nd-Laplace': {
        2: helpers.get_mechanism('2d-laplace-optimal-truncated'),
        3: helpers.get_mechanism('3d-laplace-optimal-truncated'),
        'nd': helpers.get_mechanism('nd-laplace-optimal-truncated'),
    },
    'piecewise': {
        2: helpers.get_mechanism('2d-piecewise'),
        3: helpers.get_mechanism('3d-piecewise'),
        'nd': helpers.get_mechanism('nd-piecewise'),
    }
}
model_mapper = {
    'heart-dataset': {
        2: {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=2, metric='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        },
        3: {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=4, metric='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        },
        'nd': {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3, metric='euclidean'),
            'OPTICS': OPTICS(min_samples=18, metric='euclidean')
        }
    },
    'seeds-dataset': {
        2: {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=2),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        },
        3: {
            'KMeans': KMeans(n_clusters=5, init='random', algorithm='lloyd'),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=2),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        },
        'nd': {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=4),
            'OPTICS': OPTICS(min_samples=12, metric='euclidean')
        }
    },
    'circle-dataset': {
        2: {
            'KMeans': KMeans(n_clusters=7, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=7),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        },
        3: {
            'KMeans': KMeans(n_clusters=9, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=9),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        },
        'nd': {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=2),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    }, 'line-dataset': {
        2: {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=2),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        },
        3: {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        },
        'nd': {
            'KMeans': KMeans(n_clusters=2, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=5),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    },
    'skewed-dataset': {
        2: {
            'KMeans': KMeans(n_clusters=5, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=6),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        },
        3: {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=9),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        },
        'nd': {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=4),
            #'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    }

}
datasets_mapper = {
    'heart-dataset': '../data/heart-dataset/heart_numerical.csv',
    'seeds-dataset': '../data/seeds-dataset/rq2-nd.csv',
    'circle-dataset': '../RQ3/data/circle_1000_3d.csv',
    'line-dataset': '../RQ3/data/line_1000_3d.csv',
    'skewed-dataset': '../RQ3/data/skewed_1000_3d.csv'
}
supported_datasets = ['heart', 'seeds', 'circle-dataset', 'line-dataset', 'skewed-dataset']
variants = ['nd-Laplace', 'grid-nd-Laplace', 'density-nd-Laplace', 'piecewise']

app = typer.Typer()

def plot_heatmap(df, metric, save_path=None, title=None, provided_ax=None, max_value=None, min_value=None, square=True):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax = ax if provided_ax is None else provided_ax
    scores_df = df.copy()
    prepared_df = scores_df.copy()

    if (metric == 'tpr'):
        prepared_df['tpr'] = prepared_df['tpr'].apply(
            lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x)

    prepared_df_mean = prepared_df.groupby(['epsilon', 'dimensions'])[metric].mean().reset_index()
    prepared_df_pivot = prepared_df_mean.pivot(index='dimensions', columns='epsilon', values=metric)

    heatmap = sns.heatmap(prepared_df_pivot, annot=True, robust=True, square=square, annot_kws={'fontsize':16, 'fontweight':'bold'}, fmt=".2f", linewidths=.5, ax=ax, cbar=True, cmap='Greens')
    # ax.set_title(f"TPR Scores for dataset: {dataset} with epsilon and dimensions")
    ax.set_title(title, fontsize=font_sizes['title'])
    ax.set_ylabel('Dimensions', fontsize=font_sizes['title'])
    ax.set_xlabel('Privacy budgets($\epsilon$)', fontsize=font_sizes['title'])
    heatmap.tick_params(labelsize=font_sizes['normal'])
    plt.tight_layout()
    if save_path is not None:
        #cbar = heatmap.get_children()[0]
        # plot_heatmap_legend(f'{save_path}/heatmap_legend_{metric}.png', cbar)
        # TODO: Not working?
        fig.savefig(f'{save_path}/{metric}.png', dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def get_mechanism_based_on_dimension(dimensions, variant):
    implementation = mechanism_mapper[variant]
    if dimensions in implementation:
        return implementation[dimensions]
    else:
        return implementation['nd']

def create_directory_if_nonexistent(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_color_for_cluster_algorithm(name: str):
    if name.__contains__('KMeans'):
        return 'blue'
    elif name.__contains__('Affinity'):
        return 'red'
    elif name.__contains__('OPTICS'):
        return 'green'
    elif name.__contains__('Agglomerative'):
        return 'red'

def map_mechanism_to_color(mechanism):
    if mechanism == 'density-nd-Laplace':
        return 'green'
    if mechanism == 'piecewise':
        return 'orange'
    if mechanism == 'grid-nd-Laplace':
        return 'blue'
    if mechanism == 'nd-Laplace':
        return 'red'
    else:
        return 'black'

def generate_color_palette_mechanism(algorithms):
    return {algorithm: map_mechanism_to_color(algorithm) for algorithm in algorithms}

def generate_color_palette_clustering(algorithms):
    return {algorithm: get_color_for_cluster_algorithm(algorithm) for algorithm in algorithms}

def generate_style_palette(algorithms, array):
    return {algorithm: get_style_per_algorithm(algorithm, array) for algorithm in algorithms}

def get_style_per_algorithm(algorithm, array):
    if algorithm.lower().__contains__("piecewise"):
        return array[0]
    elif algorithm.lower().__contains__("laplace"):
        return array[1]

def get_full_metric_name(metric_name):
    if metric_name == 'ari':
        return 'Adjusted Rand Index (ARI)'
    elif metric_name == 'ami':
        return 'Adjusted Mutual Information (AMI)'
    elif metric_name == 'ch':
        return 'Calinski Harabasz (CH)'
    elif metric_name == 'sc':
        return 'Silhouette score (SC)'

def calculate_baseline(models, plain_df, n_times=10):
    plain_df_scaled = StandardScaler().fit_transform(plain_df)
    plain_fitted_df = models['KMeans'].fit(plain_df_scaled)
    avg_ch = []
    avg_sh = []
    for i in range(n_times):
        ch = calinski_harabasz_score(plain_df_scaled, plain_fitted_df.labels_)
        sh = silhouette_score(plain_df_scaled, plain_fitted_df.labels_)
        avg_ch.append(ch)
        avg_sh.append(sh)

    return {'avg_ch': np.sum(avg_ch) / n_times, 'avg_sc': np.sum(avg_sh) / n_times}

def plot_results_for_mechanism_comparison(utility_metrics, compare_to, plain_df, cluster_models, export_path='../export/results/', save=True, dimension=2):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 8), linewidth=2, constrained_layout=True)

    ax_ami = plot_cluster_utility(utility_metrics, 'ami', helpers.get_experiment_epsilons(), title='', provided_ax=ax1,
                                metric=get_full_metric_name('ami'), cluster_column_name='clustering_algorithm', save=False, compare_to=compare_to)
    ax_sc =  plot_cluster_utility(utility_metrics, 'sc', helpers.get_experiment_epsilons(), title='', provided_ax=ax2,
                                metric=get_full_metric_name('sc'), cluster_column_name='clustering_algorithm', save=False, compare_to=compare_to)

    baseline = calculate_baseline(cluster_models, plain_df)
    ax_sc.axhline(y=baseline['avg_sc'], linestyle='-.', label='non-private KMeans (baseline)')

    handles, labels = ax_sc.get_legend().legend_handles, [text.get_text() for text in ax_sc.get_legend().get_texts()]
    baseline_handle = plt.Line2D([], [], color='black', linestyle='-.')
    baseline_label = 'non-private KMeans (baseline)'
    handles.append(baseline_handle)
    labels.append(baseline_label)

    ax_ami.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1))
    sns.move_legend(ax_ami, "lower center", bbox_to_anchor=(.5, 1))

    ax_sc.get_legend().remove()
    #ax_sc.get_legend().remove()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    ax_ami.set_xticks(helpers.get_experiment_epsilons())

    plt.tight_layout()
    if save:
        # Create new figure and add legend
        #figLegend = plt.figure(figsize=(1.5, 1.3))
        #plt.figlegend(*ax2.get_legend_handles_labels(), loc='upper left')

        #figLegend.savefig(f'{export_path}/legend_{dimension}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{export_path}/ami-and-sc_{dimension}_dimensions.png', dpi=300, bbox_inches='tight')
        plt.clf()


def find_baseline_mi_values(plain_df: pd.DataFrame, y_target=None, cluster_algorithm=None, n_times=10):
    plain_df_copy = plain_df.copy()
    if(y_target is None):
        create_labels = cluster_algorithm
        create_labels.fit(StandardScaler().fit_transform(plain_df_copy))
        y_target = create_labels.labels_
    else:
        plain_df_copy = plain_df_copy.drop(columns=['class'])

    targets = np.unique(y_target).size
    print('Target amount', targets);
    scores = helpers.run_mi_experiments(plain_df_copy, y_target, epsilons=[0.1], n_times=n_times,
                                        columns=plain_df_copy.columns, targets=targets)
    scores['tpr_value'] = scores['tpr'].apply(lambda x: x[1])
    scores['fpr_value'] = scores['fpr'].apply(lambda x: x[1])

    # scores['fpr'] = scores['tpr'].apply(lambda x: x.apply(lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x))
    # scores_mean = scores.mean()
    return scores['tpr_value'].mean(), scores['fpr_value'].mean(), scores['attack_adv'].mean()

def plot_cluster_utility(result, metric_name, epsilons, cluster_column_name='type',
                     metric='Adjusted Mutual Information (AMI)', provided_ax=None,
                     title='External validation (AMI & ARI): Difference between privately trained cluster algorithms '
                           'versus \n non-private trained cluster algorithms', save=True, export_file_name=None, export_path=None, compare_to=None):
    fig, ax = plt.subplots(1, sharex=True, figsize=(12, 10))
    ax_local = ax if provided_ax is None else provided_ax
    result_copy = result.copy()
    if compare_to is not None:
        compare_to_copy = compare_to.copy()
        compare_to_copy[cluster_column_name] = compare_to_copy[cluster_column_name].apply(lambda val: val + ' (Piecewise)')
        result_algorithm_string = result_copy['algorithm'].unique()
        result_copy[cluster_column_name] = result_copy[cluster_column_name].apply(lambda val: val + ' (nD-Laplace)')
        result_copy = pd.concat([result_copy, compare_to_copy], ignore_index=True)

    result_copy['Privacy mechanism'] = result_copy['algorithm']
    result_copy['Clustering algorithm'] = result_copy[cluster_column_name]
    styles = result_copy['Clustering algorithm'].unique()
    types = result_copy[cluster_column_name].unique()
    ax = sns.lineplot(x='epsilon', y=metric_name, data=result_copy, ax=ax_local, style='Clustering algorithm', markers=generate_style_palette(styles, ["s", "o"]),
                      hue='Clustering algorithm',
                      palette=generate_color_palette_clustering(types), dashes=generate_style_palette(styles, [[], [4,2]]), legend=True)


    ax_local.set_xticks(epsilons, labels=epsilons)
    ax_local.set_title(title, fontsize=font_sizes['title'])
    ax_local.set_xlabel('Privacy budget ($\epsilon$)', fontsize=font_sizes['normal'])
    ax_local.set_ylabel(metric, fontsize=font_sizes['normal'])
    ax_local.tick_params(labelsize=font_sizes['normal'])

    if save:
        file_name = metric_name if export_file_name is None else export_file_name
        fig.savefig(f'{export_path}/{file_name}.png')
        plt.clf()
    # ax.get_legend().remove()
    return ax_local

def plot_bar_colorblindness(bar):
    hatches = ['-', '+', 'x', '\\', '*', 'o', '--']
    for bars, hatch in zip(bar.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

def plot_comparison(utility_metrics: pd.DataFrame,
                    dataset,
                    tpr_baseline=None,
                    fpr_baseline=None,
                    baseline_value=None,
                    metric="Adjusted Mutual Information",
                    metric_name=None,
                    mechanism_comparison=None,
                    export_path='../export/results/',
                    title='',
                    research_question='RQ1'):
    sns.set(style="whitegrid", color_codes=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    # utility_metrics['algorithm'] = utility_metrics['algorithm'].apply(lambda x: map_mechanism_to_display_name(x))
    algorithms = utility_metrics['algorithm'].unique()

    if metric == 'tpr':
        utility_metrics['tpr'] = utility_metrics['tpr'].apply(
            lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x)
    if len(algorithms) > 1:
        bar = sns.barplot(x='epsilon', y=metric, hue="algorithm", data=utility_metrics, ax=ax,
                          palette=generate_color_palette_mechanism(algorithms))
    else:
        bar = sns.barplot(x='epsilon', y=metric, data=utility_metrics, ax=ax)

    algorithm = utility_metrics.iloc[0]['algorithm'] if mechanism_comparison is None else mechanism_comparison

    if baseline_value is not None:
        ax.axhline(y=baseline_value, linestyle='--', label='non-private K-Means (baseline)')
    if tpr_baseline is not None:
        ax.axhline(y=tpr_baseline, linestyle='solid', label=f'non-private TPR (baseline: {tpr_baseline:.2f})',
                   color='red')
    if fpr_baseline is not None:
        ax.axhline(y=fpr_baseline, linestyle='solid', label=f'non-private FPR (baseline: {fpr_baseline:.2f})',
                   color='green')
    ax.set_title(title)
    plot_bar_colorblindness(bar)
    ax.set_xlabel('Privacy Budget (epsilon)', fontsize=font_sizes['title'])
    ax.set_ylabel(metric if metric_name is None else metric_name, fontsize=font_sizes['title'])
    bar.tick_params(labelsize=font_sizes['normal'])
    bar.get_legend().remove()
    # create a second figure for the legend
    fig_leg = plt.figure(figsize=(3, 2))
    ax_leg = fig_leg.add_subplot(111)

    # draw legend from bar to the second figure
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')

    # hide the axes frame and the x/y labels
    ax_leg.axis('off')

    fig_leg.savefig(f'{export_path}/{metric}_bar_comparison_legend.png', bbox_inches='tight')
    fig.savefig(f'{export_path}/{metric}_{dataset}_comparison.png')

    plt.clf()

def run_mi_experiments(X, X_perturbed, epsilon, n_times=10, columns=['X', 'Y'], y_true_target=None, cluster_algorithm=None):
    shokri_mi_avgs = {'epsilon': [], 'shokri_mi_adv': [], 'precision': [], 'attack_adv': [], 'tpr': [], 'fpr': [], 'run': []}
    X_pd = pd.DataFrame(X, columns=columns)
    X_perturbed = X_perturbed[columns]
    y_true = y_true_target
    if y_true_target is None:
        create_labels = cluster_algorithm
        create_labels.fit(StandardScaler().fit_transform(X_pd))
        y_true = create_labels.labels_
    targets = len(np.unique(y_true))

    # create_labels = KMeans(init='random', n_clusters=4)
    # create_labels.fit(StandardScaler().fit_transform(X_pd))
    # X_pd['target'] = create_labels.labels_
    # _, _, Z = twod_laplace.generate_truncated_laplace_noise(X, epsilon)
    # Z_pd = pd.DataFrame(Z, columns=['X', 'Y'])
    # create_labels = KMeans(init='random', n_clusters=4)
    # create_labels.fit(StandardScaler().fit_transform(Z_pd))
    # target = create_labels.labels_
    for run in range(n_times):
        shokri_mi_avgs['epsilon'].append(epsilon)
        shokri_mi_avgs['run'].append(run)

        shadow_ratio = 0.75
        dataset = train_test_split(X_pd, y_true, test_size=shadow_ratio, stratify=y_true)

        x_target, x_shadow, y_target, y_shadow = dataset

        attack_train_size = len(x_target) // 2
        # attack_test_size = attack_train_size

        x_target_train = X_perturbed.iloc[x_target[:attack_train_size].index]
        x_target_train = np.array(x_target_train)
        # x_target_train = X_pd.iloc[x_target[:target_train_size].index, 0:2]
        y_target_train = y_target[:attack_train_size]
        # y_target_train = X_pd.iloc[x_target[:target_train_size].index, 2]
        x_target_test = x_target[attack_train_size:]
        y_target_test = y_target[attack_train_size:]

        # We infer based on the original data, to make sure we can estimate the dp protection
        # x_shadow_np = X_pd.iloc[x_shadow.index, 0:2].to_numpy()
        # y_shadow_np = X_pd.iloc[y_shadow.index, 2].to_numpy()
        x_shadow_np = np.array(x_shadow)
        y_shadow_np = y_shadow
        clf = RandomForestClassifier()
        print("Training shape", x_target_train.shape)
        # check if contains nan
        if np.isnan(x_target_train).any():
            print("Contains nan")
            print(x_target_train)
            break
        classifier = clf.fit(x_target_train, y_target_train)

        art_classifier = ScikitlearnRandomForestClassifier(classifier)

        ## train shadow models
        shadow_models = ShadowModels(art_classifier, num_shadow_models=3)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow_np, to_categorical(y_shadow_np, targets))
        (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

        ## Execute membership attack
        attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
        attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)

        member_infer = attack.infer(x_target[:attack_train_size], y_target_train)
        nonmember_infer = attack.infer(x_target_test, y_target_test)

        # concatenate everything and calculate roc curve
        predicted_y = np.concatenate((member_infer, nonmember_infer))
        actual_y = np.concatenate((np.ones(len(member_infer)), np.zeros(len(nonmember_infer))))
        fpr, tpr, _ = roc_curve(actual_y, predicted_y, pos_label=1)
        # attack_adv = tpr[1] / (tpr[1] + fpr[1])
        ar = recall_score(actual_y, predicted_y, pos_label=1)
        attack_advantage_yeom = ar - fpr[1]
        print('actual training', ar, attack_advantage_yeom)
        print('precision:', precision_score(actual_y, predicted_y, pos_label=1))
        print(roc_auc_score(actual_y, predicted_y))
        shokri_mi_avgs['shokri_mi_adv'].append(tpr[1] - fpr[1])
        shokri_mi_avgs['attack_adv'].append(attack_advantage_yeom)
        shokri_mi_avgs['precision'].append(precision_score(actual_y, predicted_y, pos_label=1))
        # shokri_mi_avgs['attack_adv'].append(attack_adv)
        shokri_mi_avgs['tpr'].append(tpr)
        shokri_mi_avgs['fpr'].append(fpr)

    return pd.DataFrame(shokri_mi_avgs)

def run_mi_for_dimensions(plain_df, full_perturbation_df, dataset, epsilons: list, n_times=10):
    max_columns = plain_df.drop(columns=['class']).shape[1]
    full_security_df = pd.DataFrame()
    for col in range(2, max_columns + 1):
        data = plain_df.iloc[:, 0:col]
        columns = data.columns
        y_true = plain_df['class'] if 'class' in plain_df else None
        cluster_algorithm = model_mapper[dataset]['nd' if plain_df.shape[1] > 3 else plain_df.shape[1]]['KMeans']
        for epsilon in epsilons:
            _full_perturbation_df = full_perturbation_df[
                (full_perturbation_df['dimension'] == col) & (full_perturbation_df['epsilon'] == epsilon)]
            security_df = run_mi_experiments(data, _full_perturbation_df, epsilon, columns=columns, n_times=n_times, y_true_target=y_true)
            security_df['dimension'] = col
            full_security_df = pd.concat([full_security_df, security_df], ignore_index=True)
    return full_security_df

# def plot_heatmap (dataframe: pd.DataFrame, metric: str):
def run_for_dimensions(plain_df, full_perturbation_df, mechanism, epsilon, dataset, n_times=10, model_name=None):
    dataframe = {'algorithm': [], 'clustering_algorithm': [], 'dimensions': [], 'ari': [], 'ami': [], 'ch': [],
                 'sc': [], 'mechanism': [], 'epsilon': []}
    max_columns = plain_df.drop(columns=['class']).shape[1]
    for col in range(2, max_columns + 1):
        print('Adding one column each time...')
        data = plain_df.iloc[:, 0:col]
        columns = data.columns
        print(data.shape)
        cluster_models = model_mapper[dataset]['nd' if col > 3 else col].values()

        for cluster_model in cluster_models:
            # algorithm_name = model_name if model_name is not None else helpers.map_models_to_name(cluster_model)
            dataframe['algorithm'].append(mechanism)
            dataframe['dimensions'].append(col)
            dataframe['mechanism'].append(mechanism)
            dataframe['epsilon'].append(epsilon)
            dataframe['clustering_algorithm'].append(helpers.map_models_to_name(cluster_model))
            ami_list = []
            ari_list = []
            ch_list = []
            sc_list = []
            perturbed_df = full_perturbation_df.loc[
                (full_perturbation_df['dimension'] == col) & (full_perturbation_df['epsilon'] == epsilon) & (
                            full_perturbation_df['mechanism'] == mechanism)]
            for i in range(n_times):
                perturbed_df = perturbed_df[columns]
                plain_df_scaled = StandardScaler().fit_transform(data)
                perturbed_df_scaled = StandardScaler().fit_transform(perturbed_df)
                plain_fitted_df = cluster_model.fit(plain_df_scaled)
                perturbed_fitted_df = clone(cluster_model).fit(perturbed_df_scaled)
                ami = adjusted_mutual_info_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
                ari = adjusted_rand_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
                ch = 0.0
                sc = 0
                try:
                    ch = calinski_harabasz_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
                except:
                    print('Calinski Harabasz score failed, defaulting to 0.0 as score')
                try:
                    sc = silhouette_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
                except:
                    print('Silhouette score failed, defaulting to 0 as score')

                ami_list.append(ami)
                ari_list.append(ari)
                ch_list.append(ch)
                sc_list.append(sc)

            ami = np.sum(ami_list) / n_times
            ari = np.sum(ari_list) / n_times
            sc = np.sum(sc_list) / n_times
            dataframe['ami'].append(ami)
            dataframe['ari'].append(ari)
            dataframe['ch'].append(np.sum(ch_list) / n_times)
            dataframe['sc'].append(np.sum(sc_list) / n_times)
    return pd.DataFrame(dataframe)

@app.command()
def generate_input_data(dataset: str):
    epsilons = helpers.get_experiment_epsilons()
    dataset_location = datasets_mapper[dataset]
    # TODO: Properly order this file
    plain_df = helpers.load_dataset(dataset_location)
    max_columns = plain_df.drop(columns=['class']).shape[1]
    for variant in variants:
        full_perturbation = pd.DataFrame()
        input_path = f'./data/nd-laplace/{variant}/{dataset}'
        create_directory_if_nonexistent(input_path)
        full_perturbation_loc = f'./data/nd-laplace/{variant}/{dataset}/full_perturbation.csv'
        if os.path.exists(full_perturbation_loc):
            print('Full perturbation already exists, s§ipping...')
            continue
        else:
            print('Full perturbation does not exist, generating...')
            for epsilon in epsilons:
                print(f'Running for epsilon {epsilon} and variant {variant}')
                for col in range(2, max_columns + 1):
                    print('Adding one column each time...')
                    data = plain_df.iloc[:, 0:col]
                    mechanism = get_mechanism_based_on_dimension(col, variant)
                    perturbed_df = mechanism(data, epsilon)
                    perturbed_df = pd.DataFrame(perturbed_df, columns=data.columns)
                    perturbed_df['dimension'] = col
                    perturbed_df['epsilon'] = epsilon
                    perturbed_df['mechanism'] = variant
                    full_perturbation = pd.concat([full_perturbation, perturbed_df], ignore_index=True)

            full_perturbation.to_csv(full_perturbation_loc)

@app.command()
def generate_utility_experiments(dataset: str):
    epsilons = helpers.get_experiment_epsilons()
    dataset_location = datasets_mapper[dataset]
    plain_df = helpers.load_dataset(dataset_location)
    for variant in variants:
        utility_df = pd.DataFrame()
        variant_perturbation_df_loc = f'./data/nd-laplace/{variant}/{dataset}/full_perturbation.csv'
        utility_loc = f'./data/nd-laplace/{variant}/{dataset}/utility.csv'
        variant_perturbation_df = pd.read_csv(variant_perturbation_df_loc)
        if os.path.exists(utility_loc):
            print('Utility already exists, skipping...')
            continue
        for epsilon in epsilons:
            print(f'Running for epsilon {epsilon} and variant {variant}')
            print('Utility does not exist, generating...')
            dataframe = run_for_dimensions(plain_df, variant_perturbation_df, variant, epsilon, dataset, n_times=10)
            utility_df = pd.concat([utility_df, dataframe], ignore_index=True)

        utility_df.to_csv(utility_loc)

@app.command()
def generate_distance_data(dataset: str):
    epsilons = helpers.get_experiment_epsilons()
    dataset_location = datasets_mapper[dataset]
    plain_df = helpers.load_dataset(dataset_location)
    plain_df_no_class = plain_df.drop(columns=['class'])
    max_columns = plain_df_no_class.shape[1]

    for variant in variants:
        perturbed_dataset_loc = f'./data/nd-laplace/{variant}/{dataset}/full_perturbation.csv'
        perturbed_dataset = helpers.load_dataset(perturbed_dataset_loc)
        privacy_distance_df_variant = pd.DataFrame()
        privacy_distance_loc = f'./data/nd-laplace/{variant}/{dataset}/privacy_distance.csv'
        if os.path.exists(privacy_distance_loc):
            print('Privacy distance already exists, skipping...')
            continue
        for dimension in range(2, max_columns + 1):
            print(f'Running for variant {variant} and dimension {dimension}')
            perturbed_dataset_for_dim = perturbed_dataset.loc[perturbed_dataset['dimension'] == dimension]
            data_for_dimension = plain_df_no_class.iloc[:, 0:dimension]

            privacy_distance_df = helpers.compute_euclidean_distances_between_two_datasets_per_epsilon(data_for_dimension,
                                                                                                         perturbed_dataset_for_dim,
                                                                                                       epsilons,
                                                                                                       variant,
                                                                                                       dataset,
                                                                                                       data_for_dimension.columns)
            privacy_distance_df['dimension'] = dimension
            privacy_distance_df_variant = pd.concat([privacy_distance_df_variant, privacy_distance_df], ignore_index=True)

        privacy_distance_df_variant.to_csv(privacy_distance_loc)


@app.command()
def generate_security_experiments(dataset: str):
    epsilons = helpers.get_experiment_epsilons()
    dataset_location = datasets_mapper[dataset]
    plain_df = helpers.load_dataset(dataset_location)
    for variant in variants:
        security_df = pd.DataFrame()
        variant_perturbation_df_loc = f'./data/nd-laplace/{variant}/{dataset}/full_perturbation.csv'
        if(os.path.exists(f'./data/nd-laplace/{variant}/{dataset}/security.csv')):
            print('Security already exists, skipping...')
            continue
        variant_perturbation_df = pd.read_csv(variant_perturbation_df_loc)
        dataframe = run_mi_for_dimensions(plain_df, variant_perturbation_df, dataset, epsilons, n_times=10)
        security_df = pd.concat([security_df, dataframe], ignore_index=True)

        security_df.to_csv(f'./data/nd-laplace/{variant}/{dataset}/security.csv')

@app.command()
def generate_thesis_reports(dataset: str):
    thesis_path = '/Users/tjibbevanderende/Documents/GitHub/thesis/Results'
    utility_metrics = ['ami', 'ari', 'ch', 'sc']
    dimensions = [2, 3, 'inf']
    titles = {
        'ami': 'AMI',
        'ari': 'ARI',
        'ch': 'CH',
        'sc': 'SC'
    }

    plain_df = helpers.load_dataset(datasets_mapper[dataset])
    for num_dimensions in dimensions:
        dim = 2 if num_dimensions == 2 else 3 if num_dimensions == 3 else plain_df.drop(columns=['class']).shape[1]
        for variant in variants:
            save_path = f'{thesis_path}/nd-laplace/{variant}/{dataset}'
            save_path_local = f'./data/nd-laplace/{variant}/{dataset}'
            create_directory_if_nonexistent(save_path)
            dataset_loc = f'./data/nd-laplace/{variant}/{dataset}/utility.csv'
            dataset_distance_loc = f'./data/nd-laplace/{variant}/{dataset}/privacy_distance.csv'
            dataset_df = helpers.load_dataset(dataset_loc)
            dataset_df_kmeans = dataset_df.loc[dataset_df['clustering_algorithm'].str.contains('KMeans')]
            dataset_distance_df = helpers.load_dataset(dataset_distance_loc)
            security_df_loc = f'{save_path_local}/security.csv'
            security_df = pd.read_csv(security_df_loc)
            security_df['dimensions'] = security_df['dimension']
            # dataset_df = dataset_df.loc[column_names]
            square = True if dataset in ['circle-dataset', 'line-dataset', 'skewed-dataset'] else False
            for utility_metric in utility_metrics:
                plot_heatmap(dataset_df_kmeans, utility_metric, save_path=f'{save_path_local}/',
                             title='', square=square)
                plot_heatmap(dataset_df_kmeans, utility_metric, save_path=f'{save_path}/',
                             title='', square=square)


                ## Compare cluster utility
                filter_dimensions = dataset_df['dimensions'] == dim
                dataset_cluster_utility = dataset_df.loc[(filter_dimensions) & (dataset_df['mechanism'] == variant)]
                piecewise_loc = f'./data/nd-laplace/piecewise/{dataset}/utility.csv'
                dataset_df_piecewise = helpers.load_dataset(piecewise_loc)
                dataset_df_piecewise = dataset_df_piecewise.loc[dataset_df_piecewise['dimensions'] == dim]
                #mechanism = get_mechanism_based_on_dimension(dim, variant)
                cluster_utility_export_path = f'{save_path}'
                cluster_utility_export_filename = f'cluster_utility_{utility_metric}_for_{dim}_dimensions'
                cluster_models = model_mapper[dataset]['nd' if dim > 3 else dim]
                plot_cluster_utility(dataset_cluster_utility, utility_metric, epsilons=helpers.get_experiment_epsilons(),
                                     cluster_column_name='clustering_algorithm', metric=get_full_metric_name(utility_metric),
                                     title='',
                                     export_file_name=cluster_utility_export_filename,
                                     export_path=cluster_utility_export_path, compare_to=dataset_df_piecewise)

                plot_results_for_mechanism_comparison(dataset_cluster_utility, dataset_df_piecewise, plain_df.iloc[:, 0:dim], cluster_models, cluster_utility_export_path + '/', save=True, dimension=dim)

                # export privacy distance
                dataset_distance_df['dimensions'] = dataset_distance_df['dimension']
            heatmap_min = dataset_distance_df[dataset_distance_df['algorithm'] == variant]['distance'].min()
            heatmap_max = dataset_distance_df[dataset_distance_df['algorithm'] == variant]['distance'].max()
            plot_heatmap(dataset_distance_df, 'distance', save_path=f'{save_path}/',
                         title='', square=square, min_value=heatmap_min, max_value=heatmap_max)

            #max_advantage = security_df['shokri_mi_adv'].max()
            #min_advantage = security_df['shokri_mi_adv'].min()
            ## Compare security
            plot_heatmap(security_df, 'attack_adv', save_path=f'{save_path}/', title=f'', square=square)
            plot_heatmap(security_df, 'tpr', save_path=f'{save_path}/', title=f'', square=square)

    # Compare all mechanisms
    all_utility_df = pd.DataFrame()
    all_security_df = pd.DataFrame()
    export_path = f'{thesis_path}/nd-laplace/'
    for variant in variants:
        variant_path = f'./data/nd-laplace/{variant}/{dataset}'
        variant_df = pd.read_csv(f'{variant_path}/utility.csv')
        variant_security_df = pd.read_csv(f'{variant_path}/security.csv')
        all_utility_df = pd.concat([all_utility_df, variant_df], ignore_index=True)
        variant_security_df['algorithm'] = variant
        all_security_df = pd.concat([all_security_df, variant_security_df], ignore_index=True)
    for utility_metric in utility_metrics:
        plot_comparison(all_utility_df, dataset, metric=utility_metric, export_path=f'{export_path}',)

    cluster_algorithm = model_mapper[dataset]['nd' if plain_df.shape[1] > 3 else plain_df.shape[1]]
    y_true = plain_df['class'] if 'class' in plain_df else None
    baseline_tpr, baseline_fpt, baseline_adv = find_baseline_mi_values(plain_df, y_target=y_true, n_times=10, cluster_algorithm=cluster_algorithm['KMeans'])
    plot_comparison(all_security_df, dataset, metric='attack_adv', baseline_value=baseline_adv, metric_name='Adversary advantage', export_path=f'{export_path}/')
    plot_comparison(all_security_df, dataset, metric='tpr', tpr_baseline=baseline_tpr, metric_name='True Positive Rate (TPR)', export_path=f'{export_path}/')




if __name__ == "__main__":
    app()
