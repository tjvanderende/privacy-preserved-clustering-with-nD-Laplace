from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import typer
import seaborn as sns
import os.path
from Helpers import helpers, twod_laplace, UtilityPlotter, threed_laplace, nd_laplace, ldp_mechanism, rq3_helpers
from distutils.dir_util import copy_tree

sns.color_palette("viridis", as_cmap=True)

app = typer.Typer()

research_question_1_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "2d-laplace-optimal-truncated"]
research_question_2_algorithms = ["3d-laplace", "3d-piecewise", "3d-laplace-truncated", '3d-laplace-optimal-truncated']
research_question_3_algorithms = ["nd-piecewise", "nd-laplace", "nd-laplace-truncated", "nd-laplace-optimal-truncated"]
supported_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "3d-laplace", "3d-piecewise",
                        "3d-laplace-truncated", '2d-laplace-optimal-truncated', '3d-laplace-optimal-truncated',
                        "nd-piecewise", "nd-laplace", "nd-laplace-truncated", "nd-laplace-optimal-truncated"]
supported_datasets = ["seeds-dataset", "heart-dataset"]
dataset_locations = {
    "seeds-dataset": {
        "RQ1": "../data/seeds-dataset/rq1.csv",
        "RQ2": "../data/seeds-dataset/rq2.csv",
        "RQ2-nd": "../data/seeds-dataset/rq2-nd.csv"

    },
    "heart-dataset": {
        "RQ1": "../data/heart-dataset/heart_numerical.csv",
        "RQ2": "../data/heart-dataset/heart_numerical.csv",
        "RQ2-nd": "../data/heart-dataset/heart_numerical.csv"
    }

}
dataset_algorithm_features = {
    "2d-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter"],
        "heart-dataset": ['baseline value', 'histogram_min']
    },
    "2d-piecewise": {
        "seeds-dataset": ["area", "perimeter"],
        "heart-dataset": ['baseline value', 'histogram_min']
    },
    "2d-laplace-optimal-truncated": {
        "seeds-dataset": ["area", "perimeter"],
        "heart-dataset": ['baseline value', 'histogram_min']
    },
    "2d-laplace": {
        "seeds-dataset": ["area", "perimeter"],
        "heart-dataset": ['baseline value', 'histogram_min']
    },
    "3d-laplace": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"],
        "heart-dataset": ['baseline value', 'histogram_min', 'accelerations']
    },
    "3d-piecewise": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"],
        "heart-dataset": ['baseline value', 'histogram_min', 'accelerations']
    },
    "3d-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"],
        "heart-dataset": ['baseline value', 'histogram_min', 'accelerations']
    },
    "3d-laplace-optimal-truncated": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"],
        "heart-dataset": ['baseline value', 'histogram_min', 'accelerations']
    },
    "nd-laplace": {
        "seeds-dataset": ["area", "perimeter", "compactness", "length of kernel", "width of kernel",
                          "asymmetry coefficient", "length of kernel groove"],
        "heart-dataset": ["baseline value", "accelerations", "fetal_movement", "uterine_contractions",
                          "light_decelerations", "histogram_width", "histogram_min", "histogram_max",
                          "histogram_number_of_peaks"]
    },
    "nd-piecewise": {
        "seeds-dataset": ["area", "perimeter", "compactness", "length of kernel", "width of kernel",
                          "asymmetry coefficient", "length of kernel groove"],
        "heart-dataset": ["baseline value", "accelerations", "fetal_movement", "uterine_contractions",
                          "light_decelerations", "histogram_width", "histogram_min", "histogram_max",
                          "histogram_number_of_peaks"]

    },
    "nd-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter", "compactness", "length of kernel", "width of kernel",
                          "asymmetry coefficient", "length of kernel groove"],
        "heart-dataset": ["baseline value", "accelerations", "fetal_movement", "uterine_contractions",
                          "light_decelerations", "histogram_width", "histogram_min", "histogram_max",
                          "histogram_number_of_peaks"]
    },
    "nd-laplace-optimal-truncated": {
        "seeds-dataset": ["area", "perimeter", "compactness", "length of kernel", "width of kernel",
                          "asymmetry coefficient", "length of kernel groove"],
        "heart-dataset": ["baseline value", "accelerations", "fetal_movement", "uterine_contractions",
                          "light_decelerations", "histogram_width", "histogram_min", "histogram_max",
                          "histogram_number_of_peaks"]
    }
}


def get_noise_adding_mechanism(algorithm: str, plain_df: pd.DataFrame, epsilon: float):
    mechanism = helpers.get_mechanism(algorithm)
    return mechanism(plain_df, epsilon)


def sanity_check(algorithm: str, dataset: str):
    print("Sanity check")
    if (algorithm not in supported_algorithms):
        print("Algorithm not supported")
        return;
    if (dataset not in supported_datasets):
        print("Dataset not supported")
        return;
    if (dataset not in dataset_algorithm_features[algorithm]):
        print("Dataset not supported by algorithm")
        return;


def get_export_path(dataset: str, algorithm: str, epsilon: float = None, prefix: str = 'data'):
    if (epsilon == None):
        return f"./{prefix}/{algorithm}/{dataset}/"
    else:
        return f"./{prefix}/{algorithm}/{dataset}/perturbed_{epsilon}.csv"


def get_models(dataset: str, algorithm: str):
    if (dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-laplace", "2d-piecewise",
                                                     "2d-laplace-optimal-truncated"]):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        }
    if (dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated",
                                                     "3d-laplace-optimal-truncated"]):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    if (dataset == "seeds-dataset" and algorithm in ["nd-piecewise", "nd-laplace", "nd-laplace-truncated",
                                                     "nd-laplace-optimal-truncated"]):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        }
    if (dataset == "heart-dataset" and algorithm in ['2d-laplace-truncated', '2d-laplace', '2d-piecewise',
                                                     '2d-laplace-optimal-truncated']):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        }
    if (dataset == "heart-dataset" and algorithm in ['3d-laplace-truncated', '3d-laplace', '3d-piecewise',
                                                     '3d-laplace-optimal-truncated']):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    if (dataset == "heart-dataset" and algorithm in ["nd-piecewise", "nd-laplace", "nd-laplace-truncated",
                                                     "nd-laplace-optimal-truncated"]):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=7, metric='euclidean')
        }


def get_models_for_comparison(dataset: str, algorithm: str):
    if (dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-piecewise", "2d-laplace",
                                                     "2d-laplace-optimal-truncated"]):
        return {
            '2d-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace-optimal-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd')
        }
    if (dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated",
                                                     "3d-laplace-optimal-truncated"]):
        return {
            '3d-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
        }
    if (dataset == "seeds-dataset" or "heart-dataset" and algorithm in ["nd-piecewise", "nd-laplace",
                                                                        "nd-laplace-truncated",
                                                                        "nd-laplace-optimal-truncated"]):
        return {
            'nd-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'nd-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'nd-laplace-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'nd-laplace-optimal-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd')
        }
    if (dataset == "heart-dataset" and algorithm in ['2d-laplace-truncated', '2d-laplace', '2d-piecewise',
                                                     '2d-laplace-optimal-truncated']):
        return {
            '2d-laplace-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace-optimal-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
        }


def find_baseline_values(plain_df: pd.DataFrame, model: KMeans, n_times=10):
    plain_df_scaled = StandardScaler().fit_transform(plain_df)
    ch_scores = []
    sc_scores = []
    for i in range(n_times):
        plain_fitted_df = model.fit(plain_df_scaled)
        ch = calinski_harabasz_score(plain_df_scaled, plain_fitted_df.labels_)
        sc = silhouette_score(plain_df_scaled, plain_fitted_df.labels_)
        ch_scores.append(ch)
        sc_scores.append(sc)
    return {'sc': np.mean(sc_scores), 'ch': np.mean(ch_scores)}


def find_baseline_mi_values(plain_df: pd.DataFrame, y_target, n_times=10):
    plain_df_copy = plain_df.copy()
    plain_df_copy = plain_df_copy.drop(columns=['class'])
    targets = np.unique(y_target).size
    print('Target amount', targets);
    scores = helpers.run_mi_experiments(plain_df_copy, y_target, epsilons=[0.1], n_times=n_times,
                                        columns=plain_df_copy.columns, targets=targets)
    scores['tpr_value'] = scores['tpr'].apply(lambda x: x[1])
    scores['fpr_value'] = scores['fpr'].apply(lambda x: x[1])
    # scores['fpr'] = scores['tpr'].apply(lambda x: x.apply(lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x))
    # scores_mean = scores.mean()
    return scores['tpr_value'].mean(), scores['fpr_value'].mean()


def plot_bar_colorblindness(bar):
    hatches = ['-', '+', 'x', '\\', '*', 'o', '--']
    for bars, hatch in zip(bar.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)


def plot_dimension_comparison(metrics: pd.DataFrame, dataset: str, filename: str, metric='ami', ):
    sns.set(style="whitegrid", color_codes=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    bar = sns.barplot(x='dimensions', y=metric, hue="type", data=metrics)
    ax.set_title(f"Dimension comparison of {metric} for {dataset} using K-Means.")
    plot_bar_colorblindness(bar)
    ax.legend(title='Mechanism')
    fig.savefig(f'results/RQ3/{dataset}/{filename}.png')
    plt.clf()


def plot_comparison(utility_metrics: pd.DataFrame,
                    dataset,
                    algorithm_type,
                    tpr_baseline=None,
                    fpr_baseline=None,
                    baseline_value=None,
                    metric="Adjusted Mutual Information",
                    mechanism_comparison=None,
                    research_question='RQ1'):
    sns.set(style="whitegrid", color_codes=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    bar = sns.barplot(x='epsilon', y=metric, hue="algorithm", data=utility_metrics, ax=ax)
    algorithm = utility_metrics.iloc[0]['algorithm'] if mechanism_comparison is None else mechanism_comparison
    features = dataset_algorithm_features[algorithm_type][dataset]
    if baseline_value is not None:
        ax.axhline(y=baseline_value, linestyle='--', label='non-private K-Means (baseline)')
    if tpr_baseline is not None and fpr_baseline is not None:
        ax.axhline(y=tpr_baseline, linestyle='solid', label=f'non-private TPR (baseline: {tpr_baseline:.2f})',
                   color='red')
        ax.axhline(y=fpr_baseline, linestyle='solid', label=f'non-private FPR (baseline: {fpr_baseline:.2f})',
                   color='green')
    ax.set_title(
        f"Comparison of {metric} for {dataset} using K-Means. Algorithm: {algorithm}. Dimensions: {len(features)}")
    plot_bar_colorblindness(bar)

    ax.legend(title='Mechanism')
    fig.savefig('results/' + research_question + '/' + dataset + '/' + metric + '_' + dataset + '_comparison.png')
    plt.clf()


def plot_tpr_fpr_comparison(private_dataset: pd.DataFrame, dataset, research_question='RQ1', metric='Shokri MI'):
    sns.set(style="whitegrid", color_codes=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    prepared_data = helpers.prepare_for_roc(private_dataset)
    private_dataset['tpr'] = prepared_data['tpr']
    private_dataset['fpr'] = prepared_data['fpr']
    private_dataset.drop(columns=['run', 'shokri_mi_adv', 'attack_adv'], inplace=True)

    df_stacked = private_dataset
    df_stacked['complement'] = 1

    # Create the grouped bar plot using seaborn
    sns.barplot(x='epsilon', y='tpr', hue='algorithm',
                data=df_stacked, ci=None, palette='colorblind')

    # Plot the complement bars
    sns.barplot(x='epsilon', y='complement', hue='algorithm',
                data=df_stacked, ci=None, palette='colorblind', alpha=0.3)

    # private_dataset.groupby(['epsilon', 'algorithm']).mean().plot(kind='bar', stacked=True, y=['tpr', 'fpr'], ax=ax)
    # bar2 = sns.barplot(x='epsilon', y='fpr', hue="algorithm", data=prepared_data, ax=ax, alpha=0.5)
    # bar = sns.barplot(x='epsilon', y='tpr', hue="algorithm", data=prepared_data, ax=ax)

    ax.legend(title='Mechanism')
    fig.savefig('results/' + research_question + '/' + dataset + '/' + metric + '_' + dataset + '_comparison_rate.png')
    plt.clf()


def create_directory_if_nonexistent(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_algorithm_prefx(algorithm):
    return algorithm.replace('nd-', '', regex=True).replace('2d-', '', regex=True).replace('3d-', '', regex=True)


def export_for_report(comparison_security: pd.DataFrame, dataset: str, research_question: str,
                      baseline_value: float = None):
    ## Export tables ##
    comparison_security.rename(
        columns={'shokri_mi_adv': 'Shokri MI', 'attack_adv': 'Attack advantage', 'tpr': 'True Positive Rate',
                 'fpr': 'False Positive Rate'}, inplace=True)
    comparison_security['algorithm'] = remove_algorithm_prefx(comparison_security['algorithm'])
    grouped = comparison_security.groupby(['algorithm', 'epsilon']).mean().round(3)
    grouped.drop(columns=['complement'], inplace=True)
    if baseline_value is not None:
        grouped = grouped[grouped["True Positive Rate"].ge(baseline_value)]
    grouped.to_csv(get_export_path(dataset, research_question, prefix='results') + 'privacy_scores_mean.csv')
    grouped.to_latex(get_export_path(dataset, research_question, prefix='results') + 'privacy_scores_mean.tex')


@app.command()
def run_utility_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running utility experiments on {plain_dataset_path} with {algorithm}")
    plain_df = helpers.load_dataset(plain_dataset_path)
    epsilons = helpers.get_experiment_epsilons()
    sanity_check(algorithm, dataset)
    print('run experiments for: ', epsilons)
    features = dataset_algorithm_features[algorithm][dataset]
    # plain_df[features] = helpers.reshape_data_to_uniform(plain_df[features])

    print(plain_df.head())
    create_directory_if_nonexistent(get_export_path(dataset, algorithm))
    # --- RUN PERTURBATION ALGORITHM ---
    for epsilon in epsilons:
        Z = get_noise_adding_mechanism(algorithm, plain_df[features], epsilon)
        Z_pd = pd.DataFrame(Z, columns=features)
        Z_pd.to_csv(get_export_path(dataset, algorithm, epsilon), index=False)

    supported_models = list(get_models(dataset, algorithm).values())
    print('Generated report for: ', supported_models)

    # --- RUN CLUSTERING ALGORITHMS ---
    create_directory_if_nonexistent(get_export_path(dataset, algorithm, prefix='results'))
    utility_dataset = get_export_path(dataset, algorithm, prefix='results') + 'utility_scores.csv';
    if os.path.isfile(utility_dataset):
        print('Loading existing utility report')
        report = helpers.load_dataset(utility_dataset)
    else:
        print('Generating new utility report')
        report = helpers.generate_external_validity_export(
            epsilons,
            supported_models,
            import_path=plain_dataset_path,
            perturbed_path=get_export_path(dataset, algorithm), columns=features)
        report.to_csv(get_export_path(dataset, algorithm, prefix='results') + 'utility_scores.csv', index=False)

    # --- PLOT RESULTS ---
    utility = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm), columns=features)
    utility.plot_external_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)
    utility_internal = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm),
                                                     columns=features)
    utility_internal.plot_internal_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)


@app.command()
def run_comparison_experiment(research_question: str, dataset: str):
    print(f'Running {research_question}')
    algorithm = '2d-laplace-truncated' if research_question == 'RQ1' else '3d-laplace' if research_question == 'RQ2' else 'nd-laplace'
    algorithms = research_question_1_algorithms if research_question == 'RQ1' else research_question_2_algorithms if research_question == 'RQ2' else research_question_3_algorithms
    algorithm_model = get_models(dataset=dataset, algorithm=algorithm)['KMeans']
    model_name = helpers.map_models_to_name(algorithm_model)

    print('Considering:', model_name)
    print('Algorithms', algorithms)
    for dataset in supported_datasets:
        comparison_dp = pd.DataFrame()
        comparison_dp_security = pd.DataFrame()
        comparison_dp_security_distance = pd.DataFrame()
        for algorithm in algorithms:
            create_directory_if_nonexistent(get_export_path(dataset, research_question, prefix='results'))

            # external_laplace = helpers.load_dataset(get_export_path('seeds-dataset', '2d-laplace-truncated', prefix='results')+'utility_scores.csv')
            ultility_metrics = helpers.load_dataset(
                get_export_path(dataset, algorithm, prefix='results') + '/utility_scores.csv')
            ultility_metrics = ultility_metrics[ultility_metrics['type'] == model_name]
            ultility_metrics['algorithm'] = algorithm

            print(dataset, algorithm)
            privacy_metrics = helpers.load_dataset(
                get_export_path(dataset, algorithm, prefix='results') + '/privacy_scores.csv')
            privacy_metrics['algorithm'] = algorithm

            privacy_distance_dataset = helpers.load_dataset(
                get_export_path(dataset, algorithm, prefix='results') + 'privacy_distance_scores.csv')

            # utility_dimensionality_dataset = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results') + 'utility_dimensionality_scores.csv')

            comparison_dp = pd.concat([comparison_dp, ultility_metrics]).reset_index(drop=True)
            comparison_dp_security = pd.concat([comparison_dp_security, privacy_metrics]).reset_index(drop=True)
            comparison_dp_security_distance = pd.concat(
                [comparison_dp_security_distance, privacy_distance_dataset]).reset_index(drop=True)

        """
        Load baseline for membership inference attack
        """
        dataset_loc = dataset_locations[dataset][research_question]
        plain_df = helpers.load_dataset(dataset_loc)
        y_target = plain_df['class']
        tpr_baseline, fpr_baseline = find_baseline_mi_values(plain_df, y_target, n_times=50)

        plot_comparison(comparison_dp_security,
                        dataset,
                        algorithm,
                        metric='shokri_mi_adv',
                        mechanism_comparison='Adversary advantage',
                        research_question=research_question)
        plot_tpr_fpr_comparison(comparison_dp_security, dataset, research_question=research_question)
        roc_plot_title = 'ROC plot on ' + dataset + '/n with shape: ' + str(ultility_metrics.shape)

        helpers.display_roc_plot(
            comparison_dp_security,
            algorithms,
            title=roc_plot_title,
            tpr_baseline=tpr_baseline,
            fpr_baseline=fpr_baseline,
            save_as=get_export_path(dataset, research_question, prefix='results') + 'roc_plot.png')

        export_for_report(comparison_dp_security, dataset, research_question, baseline_value=tpr_baseline)

        helpers.create_lineplot_of_different_algorithms(
            comparison_dp_security_distance,
            f"Difference in euclidean distance between non-private and private variant of the {dataset} for each mechanism.",
            "Epsilon",
            "Euclidean distance",
            safe_path=get_export_path(dataset, research_question, prefix='results') + 'privacy_distance_plot.png')

        for metric in ['ami', 'ari', 'ch', 'sc']:
            baseline_value = None
            if metric == 'ch' or metric == 'sc':
                baseline_value = find_baseline_values(plain_df, algorithm_model)[metric]
                print('baseline value', baseline_value)
            plot_comparison(comparison_dp, dataset, algorithm, metric=metric, research_question=research_question,
                            baseline_value=baseline_value)

    print("Concatenated utility results:", comparison_dp.head())
    print("Concatenated security results:", comparison_dp_security.head())


@app.command()
def run_privacy_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running privacy experiments on {plain_dataset_path} with {algorithm}")
    sanity_check(algorithm, dataset)

    plain_df = helpers.load_dataset(plain_dataset_path)
    y_target = plain_df['class']
    X_features = plain_df[dataset_algorithm_features[algorithm][dataset]]
    # X_features = helpers.reshape_data_to_uniform(X_features)

    print('Features: ', X_features.head())
    print('Target', y_target.head())
    epsilons = helpers.get_experiment_epsilons()
    print('run experiments for: ', epsilons)
    targets = np.unique(y_target).size
    print('Target amount', targets);

    privacy_dataset = get_export_path(dataset, algorithm, prefix='results') + 'privacy_scores.csv';
    privacy_distance_dataset = get_export_path(dataset, algorithm, prefix='results') + 'privacy_distance_scores.csv';
    if os.path.isfile(privacy_distance_dataset):
        print('Loading existing distance report')
    else:
        print(X_features.head(), algorithm, dataset)
        privacy_distance_df = helpers.compute_euclidean_distances_between_two_datasets_per_epsilon(X_features, epsilons,
                                                                                                   algorithm, dataset)
        privacy_distance_df.to_csv(
            get_export_path(dataset, algorithm, prefix='results') + 'privacy_distance_scores.csv', index=False)

    if os.path.isfile(privacy_dataset):
        print('Loading existing report')
        # report = helpers.load_dataset(privacy_dataset)
    else:
        # --- RUN EXPERIMENTS ---
        privacy_df = helpers.run_mi_experiments(X_features.values, y_target.values, epsilons,
                                                algorithm=helpers.get_mechanism(algorithm), n_times=50,
                                                columns=dataset_algorithm_features[algorithm][dataset],
                                                targets=targets);
        privacy_df.to_csv(get_export_path(dataset, algorithm, prefix='results') + 'privacy_scores.csv', index=False)


@app.command()
def run_experiments_rq3():
    for dataset in supported_datasets:
        ## DATASET LEVEL ##
        plain_dataset_location = dataset_locations[dataset]['RQ2-nd']

        k_means_model = get_models(dataset, 'nd-laplace')['KMeans']
        plain_dataset = helpers.load_dataset(plain_dataset_location)
        plain_dataset_with_target = plain_dataset.copy()
        plain_dataset.drop(columns=['class'], inplace=True)
        epsilons = [0.5, 3, 9]
        n_times_per_epsilon_for_mi = 50
        """
        Run RQ3 things
        """
        for epsilon in epsilons:

            utility_dimensional_loc = f'./results/RQ3/{dataset}/utility_dimensionality_scores_{epsilon}.csv'
            security_dimensional_loc = f'./results/RQ3/{dataset}/security_dimensionality_scores.csv'
            if (os.path.isfile(utility_dimensional_loc)):
                print(f'Use existing utility dimensional dataset for {epsilon}')
                utility_dimensions = helpers.load_dataset(utility_dimensional_loc)
            else:
                utility_dimensions = rq3_helpers.run_for_dimensions_and_algorithms(
                    plain_dataset,
                    epsilon,
                    [k_means_model],
                    research_question_3_algorithms,
                    dataset=dataset,
                    import_path=plain_dataset_location,
                    perturbed_path=f'./data/',
                )
                create_directory_if_nonexistent(f'./results/RQ3/{dataset}/')
                utility_dimensions.to_csv(utility_dimensional_loc, index=False)

            ## PLOT DIMENSIONALITY (Utility) ##
            plot_dimension_comparison(utility_dimensions, dataset, f'utility_dimensions_{epsilon}', metric='ami')

        if os.path.isfile(security_dimensional_loc):
            print(f'Use existing security dimensional dataset for multiple epsilons')
            security_dimensions = helpers.load_dataset(security_dimensional_loc)
        else:
            security_dimensions = rq3_helpers.run_security_mi_for_dimensions_and_algorithm(
                plain_dataset_with_target,
                ['nd-laplace-optimal-truncated', 'nd-piecewise'],
                epsilons,
                n_times=n_times_per_epsilon_for_mi,
                target_column='class'
            )
            security_dimensions.to_csv(security_dimensional_loc, index=False)

        ## PLOT DIMENSIONALITY (Security) ##
        # plot_dimension_comparison(security_dimensions, dataset, f'security_dimensions', metric='ami')


if __name__ == "__main__":
    app()
