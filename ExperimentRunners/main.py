from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation, KMeans
import typer
import seaborn as sns
import os.path
from Helpers import helpers, twod_laplace, UtilityPlotter, threed_laplace, nd_laplace, ldp_mechanism
from distutils.dir_util import copy_tree

sns.color_palette("viridis", as_cmap=True)

app = typer.Typer()

research_question_1_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "2d-laplace-optimal-truncated"]
research_question_2_algorithms = ["3d-laplace", "3d-piecewise", "3d-laplace-truncated", '3d-laplace-optimal-truncated']
research_question_3_algorithms = ["nd-piecewise", "nd-laplace", "nd-laplace-truncated"]
supported_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "3d-laplace", "3d-piecewise", "3d-laplace-truncated", '2d-laplace-optimal-truncated', '3d-laplace-optimal-truncated', "nd-piecewise", "nd-laplace", "nd-laplace-truncated"]
supported_datasets = ["seeds-dataset", "heart-dataset"]
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
        "seeds-dataset": ["area","perimeter","compactness","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove"],
        "heart-dataset": ["baseline value","accelerations","fetal_movement","uterine_contractions","light_decelerations","histogram_width","histogram_min","histogram_max","histogram_number_of_peaks"]
    },
    "nd-piecewise": {
        "seeds-dataset": ["area","perimeter","compactness","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove"],
        "heart-dataset": ["baseline value","accelerations","fetal_movement","uterine_contractions","light_decelerations","histogram_width","histogram_min","histogram_max","histogram_number_of_peaks"]

    },
    "nd-laplace-truncated": {
        "seeds-dataset": ["area","perimeter","compactness","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove"],
        "heart-dataset": ["baseline value","accelerations","fetal_movement","uterine_contractions","light_decelerations","histogram_width","histogram_min","histogram_max","histogram_number_of_peaks"]
    }
}

def get_mechanism(algorithm):
    mechanism = ldp_mechanism.ldp_mechanism()
    if(algorithm == "2d-laplace-truncated"):
        return twod_laplace.generate_truncated_laplace_noise
    if(algorithm == "2d-piecewise"):
        return helpers.generate_piecewise_perturbation
    if(algorithm == "2d-laplace"):
        return twod_laplace.generate_laplace_noise_for_dataset
    if(algorithm == "2d-laplace-optimal-truncated"):
        return mechanism.randomise
    if(algorithm == "3d-laplace"):
        return threed_laplace.generate_3D_noise_for_dataset
    if(algorithm == "3d-piecewise"):
        return helpers.generate_piecewise_perturbation
    if(algorithm == "3d-laplace-truncated"):
        return threed_laplace.generate_truncated_perturbed_dataset
    if(algorithm == "3d-laplace-optimal-truncated"):
        return mechanism.randomise
    if(algorithm == "nd-laplace-truncated"):
        return nd_laplace.generate_nd_laplace_noise_for_dataset
    if(algorithm == "nd-laplace"):
        return nd_laplace.generate_nd_laplace_noise_for_dataset;
    if(algorithm == "nd-piecewise"):
        return helpers.generate_piecewise_perturbation;
    if(algorithm == "nd-laplace-truncated"):
        return nd_laplace.generate_truncated_nd_laplace_noise_for_dataset;

def get_noise_adding_mechanism(algorithm: str, plain_df: pd.DataFrame, epsilon: float):
    mechanism = get_mechanism(algorithm)
    return mechanism(plain_df, epsilon)
    

def sanity_check(algorithm: str, dataset: str):
    print("Sanity check")
    if(algorithm not in supported_algorithms):
        print("Algorithm not supported")
        return;
    if(dataset not in supported_datasets):
        print("Dataset not supported")
        return;
    if(dataset not in dataset_algorithm_features[algorithm]):
        print("Dataset not supported by algorithm")
        return;

def get_export_path(dataset: str, algorithm: str, epsilon: float = None, prefix: str = 'data'):
    if(epsilon == None):
        return f"./{prefix}/{algorithm}/{dataset}/"
    else:
        return f"./{prefix}/{algorithm}/{dataset}/perturbed_{epsilon}.csv"

def get_models(dataset: str, algorithm: str):
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-laplace", "2d-piecewise", "2d-laplace-optimal-truncated"]):
        return {
           'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        }
    if(dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated", "3d-laplace-optimal-truncated"]):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    if(dataset == "seeds-dataset" and algorithm in ["nd-piecewise", "nd-laplace", "nd-laplace-truncated"]):
        return {
           'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=7, metric='euclidean')
        }
    if(dataset == "heart-dataset" and algorithm in ['2d-laplace-truncated', '2d-laplace', '2d-piecewise', '2d-laplace-optimal-truncated']):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=4, metric='euclidean')
        }
    if(dataset == "heart-dataset" and algorithm in ['3d-laplace-truncated', '3d-laplace', '3d-piecewise', '3d-laplace-optimal-truncated']):
        return {
            'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'OPTICS': OPTICS(min_samples=6, metric='euclidean')
        }
    if(dataset == "heart-dataset" and algorithm in ["nd-piecewise", "nd-laplace", "nd-laplace-truncated"]):
        return {
           'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            # 'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
           'OPTICS': OPTICS(min_samples=7, metric='euclidean')
        }
def get_models_for_comparison(dataset: str, algorithm: str):
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "2d-laplace-optimal-truncated"]):
        return {
           '2d-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
           '2d-laplace-truncated':KMeans(n_clusters=4, init='random', algorithm='lloyd'),
           '2d-laplace-optimal-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd')
        }
    if(dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated", "3d-laplace-optimal-truncated"]):
        return {
           '3d-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
        }
    if(dataset == "seeds-dataset" or "heart-dataset" and algorithm in ["nd-piecewise", "nd-laplace", "nd-laplace-truncated"]):
        return {
           'nd-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
           'nd-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'nd-laplace-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd')
        }
    if(dataset == "heart-dataset" and algorithm in ['2d-laplace-truncated', '2d-laplace', '2d-piecewise', '2d-laplace-optimal-truncated']):
        return {
            '2d-laplace-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-piecewise': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            '2d-laplace-optimal-truncated': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
        }
    
def plot_comparison(utility_metrics: pd.DataFrame, dataset, algorithm_type, metric = "ami", mechanism_comparison = None, research_question = 'RQ1'): 
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='epsilon', y=metric, hue="algorithm", data=utility_metrics, ax=ax, errorbar=None)
    algorithm = utility_metrics.iloc[0]['algorithm'] if mechanism_comparison is None else mechanism_comparison
    features = dataset_algorithm_features[algorithm_type][dataset]
    ax.set_title(f"Comparison of {metric} for {dataset}. Algorithm: {algorithm}. Dimensions: {len(features)}")
    fig.savefig('results/'+research_question+'/' + dataset + '/' +metric+'_'+dataset+'_comparison.png')
    plt.clf()

def create_directory_if_nonexistent(path):
    if not os.path.exists(path):
        os.makedirs(path)

@app.command()
def run_utility_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running utility experiments on {plain_dataset_path} with {algorithm}")
    plain_df = helpers.load_dataset(plain_dataset_path)
    epsilons = helpers.get_experiment_epsilons()
    sanity_check(algorithm, dataset)
    print('run experiments for: ', epsilons)
    features = dataset_algorithm_features[algorithm][dataset]
    #plain_df[features] = helpers.reshape_data_to_uniform(plain_df[features])

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
    utility_dataset = get_export_path(dataset, algorithm, prefix='results')+'utility_scores.csv';
    report = pd.DataFrame();
    if os.path.isfile(utility_dataset):
        print('Loading existing report')
        report = helpers.load_dataset(utility_dataset)
    else: 
        print('Generating new report')
        report = helpers.generate_external_validity_export(
            epsilons, 
            supported_models,
            import_path=plain_dataset_path, 
            perturbed_path=get_export_path(dataset, algorithm), columns=features)
        report.to_csv(get_export_path(dataset, algorithm, prefix='results')+'utility_scores.csv', index=False)

    # --- PLOT RESULTS ---
    utility = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm), columns=features)
    utility.plot_external_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)
    utility_internal = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm), columns=features)
    utility_internal.plot_internal_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)

@app.command()
def run_comparison_experiment(research_question: str, dataset: str):
    print(f'Running {research_question}')
    algorithm = '2d-laplace-truncated' if research_question == 'RQ1' else '3d-laplace' if research_question == 'RQ2' else 'nd-laplace'
    algorithms = research_question_1_algorithms if research_question == 'RQ1' else research_question_2_algorithms if research_question == 'RQ2' else research_question_3_algorithms
    model_name = helpers.map_models_to_name(get_models(dataset=dataset, algorithm=algorithm)['KMeans'])

    print('Considering:', model_name)       
    print('Algorithms', algorithms)
    for dataset in supported_datasets:
        comparison_dp = pd.DataFrame()
        comparison_dp_security = pd.DataFrame()
        for algorithm in algorithms:
            create_directory_if_nonexistent(get_export_path(dataset, research_question, prefix='results'))

            #external_laplace = helpers.load_dataset(get_export_path('seeds-dataset', '2d-laplace-truncated', prefix='results')+'utility_scores.csv')
            ultility_metrics = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results')+'/utility_scores.csv')
            ultility_metrics = ultility_metrics[ultility_metrics['type'] == model_name]
            ultility_metrics['algorithm'] = algorithm

            print(dataset, algorithm)
            privacy_metrics = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results')+'/privacy_scores.csv')
            privacy_metrics['algorithm'] = algorithm            


            comparison_dp = pd.concat([comparison_dp, ultility_metrics]).reset_index(drop=True)
            comparison_dp_security = pd.concat([comparison_dp_security, privacy_metrics]).reset_index(drop=True)

        ## DATASET LEVEL ## 
        plot_comparison(comparison_dp_security, dataset, algorithm, metric='shokri_mi_adv', mechanism_comparison='Adversary advantage', research_question=research_question)    
        roc_plot_title = 'ROC plot on '+ dataset + '/n with shape: ' + str(ultility_metrics.shape)
        helpers.display_roc_plot(comparison_dp_security, algorithms, title=roc_plot_title, save_as = get_export_path(dataset, research_question, prefix='results')+'roc_plot.png')
        for metric in ['ami', 'ari', 'ch', 'sc']:
            plot_comparison(comparison_dp, dataset, algorithm, metric=metric, research_question=research_question)

    print("Concatenated utility results:", comparison_dp.head())
    print("Concatenated security results:", comparison_dp_security.head())

 
@app.command()
def run_privacy_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running privacy experiments on {plain_dataset_path} with {algorithm}")
    sanity_check(algorithm, dataset)

    plain_df = helpers.load_dataset(plain_dataset_path)
    y_target = plain_df['class']
    X_features = plain_df[dataset_algorithm_features[algorithm][dataset]]
    #X_features = helpers.reshape_data_to_uniform(X_features)

    print('Features: ', X_features.head())
    print('Target', y_target.head())
    epsilons = helpers.get_experiment_epsilons()
    print('run experiments for: ', epsilons)
    targets = np.unique(y_target).size
    print('Target amount', targets);
    # --- RUN EXPERIMENTS ---
    privacy_df = helpers.run_mi_experiments(X_features.values, y_target.values, epsilons, algorithm=get_mechanism(algorithm), n_times=50, columns=dataset_algorithm_features[algorithm][dataset], targets=targets);
    privacy_df.to_csv(get_export_path(dataset, algorithm, prefix='results')+'privacy_scores.csv', index=False)
        

    
if __name__ == "__main__":
    app()