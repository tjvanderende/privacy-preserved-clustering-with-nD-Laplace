from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
import typer
import seaborn as sns
import os.path
from Helpers import helpers, twod_laplace, UtilityPlotter, threed_laplace
from distutils.dir_util import copy_tree

sns.color_palette("viridis", as_cmap=True)

app = typer.Typer()

research_question_1_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace"]
research_question_2_algorithms = ["3d-laplace", "3d-piecewise", "3d-laplace-truncated"]
research_question_3_algorithms = ["piecewise", "nd-laplace"]
supported_algorithms = ["2d-laplace-truncated", "2d-piecewise", "2d-laplace", "3d-laplace", "3d-piecewise", "3d-laplace-truncated", "piecewise", "nd-laplace"]
supported_datasets = ["seeds-dataset"]
dataset_algorithm_features = {
    "2d-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter"],
    },
    "2d-piecewise": {
        "seeds-dataset": ["area", "perimeter"],
    },
    "2d-laplace": {
        "seeds-dataset": ["area", "perimeter"],
    },
    "3d-laplace": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"]
    },
    "3d-piecewise": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"]
    },
    "3d-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter", "length of kernel"]
    },
    "nd-laplace": {
        "seeds-dataset": ["area","perimeter","compactness","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove"]
    },
    "piecewice": {
        "seeds-dataset": ["area","perimeter","compactness","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove"]
    }
}

def get_mechanism(algorithm):
    if(algorithm == "2d-laplace-truncated"):
        return twod_laplace.generate_truncated_laplace_noise
    if(algorithm == "2d-piecewise"):
        return helpers.generate_piecewise_perturbation
    if(algorithm == "2d-laplace"):
        return twod_laplace.generate_laplace_noise_for_dataset
    if(algorithm == "3d-laplace"):
        return threed_laplace.generate_3D_noise_for_dataset
    if(algorithm == "3d-piecewise"):
        return helpers.generate_piecewise_perturbation
    if(algorithm == "3d-laplace-truncated"):
        return threed_laplace.generate_truncated_perturbed_dataset
    if(algorithm == "nd-laplace"):
        return;
    if(algorithm == "piecewise"):
        return;

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
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-piecewise", "2d-laplace"]):
        return {
           'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'DBSCAN': DBSCAN(min_samples=4, metric='euclidean', eps=0.3)
        }
    if(dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated"]):
        return {
           'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'DBSCAN': DBSCAN(min_samples=6, metric='euclidean', eps=0.5)
        }
    

def get_models_for_comparison(dataset: str, algorithm: str):
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-piecewise", "2d-laplace"]):
        return {
           '2d-piecewise': KMeans(n_clusters=3, init='random', algorithm='lloyd'),
           '2d-laplace-truncated':KMeans(n_clusters=3, init='random', algorithm='lloyd')
        }
    if(dataset == "seeds-dataset" and algorithm in ["3d-laplace", "3d-piecewise", "3d-laplace-truncated"]):
        return {
           '3d-laplace': KMeans(n_clusters=3, init='random', algorithm='lloyd'),
        }
    
    
def plot_comparison(utility_metrics: pd.DataFrame, dataset, metric = "ami", mechanism_comparison = None, research_question = 'RQ1'): 
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='epsilon', y=metric, hue="algorithm", data=utility_metrics, ax=ax)
    algorithm = utility_metrics.iloc[0]['algorithm'] if mechanism_comparison is None else mechanism_comparison
    ax.set_title(f"Comparison of {metric} for {dataset}. Algorithm: {algorithm}")
    fig.savefig('results/'+research_question+'/' + dataset + '/' +metric+'_'+dataset+'_comparison.png')

@app.command()
def run_utility_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running utility experiments on {plain_dataset_path} with {algorithm}")
    plain_df = helpers.load_dataset(plain_dataset_path)
    epsilons = helpers.get_experiment_epsilons()
    sanity_check(algorithm, dataset)
    print('run experiments for: ', epsilons)
    features = dataset_algorithm_features[algorithm][dataset]
    plain_df[features] = helpers.reshape_data_to_uniform(plain_df[features])

    print(plain_df.head())
    # --- RUN PERTURBATION ALGORITHM ---
    for epsilon in epsilons:    
        Z = get_noise_adding_mechanism(algorithm, plain_df[features], epsilon)
        Z_pd = pd.DataFrame(Z, columns=features)
        Z_pd.to_csv(get_export_path(dataset, algorithm, epsilon), index=False)
        
    supported_models = list(get_models(dataset, algorithm).values())
    print('Generated report for: ', supported_models)

    # --- RUN CLUSTERING ALGORITHMS ---
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
def run_comparison_experiment(research_question: str):
    print(f'Running {research_question}')
    algorithm = '2d-laplace-truncated' if research_question == 'RQ1' else '3d-laplace'
    algorithms = research_question_1_algorithms if research_question == 'RQ1' else research_question_2_algorithms
    model_name = helpers.map_models_to_name(get_models(dataset='seeds-dataset', algorithm=algorithm)['KMeans'])
    print('Considering:', model_name)       
    comparison_dp = pd.DataFrame()
    comparison_dp_security = pd.DataFrame()
    for algorithm in algorithms:
        for dataset in supported_datasets:
            #external_laplace = helpers.load_dataset(get_export_path('seeds-dataset', '2d-laplace-truncated', prefix='results')+'utility_scores.csv')
            ultility_metrics = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results')+'/utility_scores.csv')
            ultility_metrics = ultility_metrics[ultility_metrics['type'] == model_name]
            ultility_metrics['algorithm'] = algorithm

            privacy_metrics = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results')+'/privacy_scores.csv')
            privacy_metrics['algorithm'] = algorithm

            #print(ultility_metrics.head())
            comparison_dp = pd.concat([comparison_dp, ultility_metrics]).reset_index(drop=True)
            comparison_dp_security = pd.concat([comparison_dp_security, privacy_metrics]).reset_index(drop=True)

    print("Concatenated utility results:", comparison_dp.head())
    print("Concatenated security results:", comparison_dp_security.head())

    for dataset in supported_datasets:
        for metric in ['ami', 'ari', 'ch', 'sc']:
            plot_comparison(comparison_dp, dataset, metric=metric, research_question=research_question)
        plot_comparison(comparison_dp_security, dataset, metric='shokri_mi_adv', mechanism_comparison='Adversary advantage', research_question=research_question)

@app.command()
def run_privacy_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running privacy experiments on {plain_dataset_path} with {algorithm}")
    sanity_check(algorithm, dataset)

    plain_df = helpers.load_dataset(plain_dataset_path)
    y_target = plain_df['class']
    X_features = plain_df[dataset_algorithm_features[algorithm][dataset]]
    X_features = helpers.reshape_data_to_uniform(X_features)

    print('Features: ', X_features.head())
    print('Target', y_target.head())
    epsilons = helpers.get_experiment_epsilons()
    print('run experiments for: ', epsilons)
    
    # --- RUN EXPERIMENTS ---
    privacy_df = helpers.run_mi_experiments(X_features.values, y_target.values, epsilons, algorithm=get_mechanism(algorithm), columns=dataset_algorithm_features[algorithm][dataset], targets=3);
    privacy_df.to_csv(get_export_path(dataset, algorithm, prefix='results')+'privacy_scores.csv', index=False)
        

    
if __name__ == "__main__":
    app()