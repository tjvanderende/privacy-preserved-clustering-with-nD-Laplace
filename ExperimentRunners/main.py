from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
import typer
import seaborn as sns
from Helpers import helpers, twod_laplace, UtilityPlotter
app = typer.Typer()

research_question_1_algorithms = ["2d-laplace-truncated", "2d-pairwise"]
supported_algorithms = ["2d-laplace-truncated", "2d-pairwise"]
supported_datasets = ["seeds-dataset"]
dataset_algorithm_features = {
    "2d-laplace-truncated": {
        "seeds-dataset": ["area", "perimeter"]
    },
    "2d-pairwise": {
        "seeds-dataset": ["area", "perimeter"]
    }
}

def get_noise_adding_mechanism(algorithm: str, plain_df: pd.DataFrame, epsilon: float):
    if algorithm == "2d-laplace-truncated":
        return twod_laplace.generate_truncated_laplace_noise(plain_df, epsilon)
    if algorithm == "2d-pairwise":
        return helpers.generate_pairwise_perturbation(plain_df, epsilon)

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
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-pairwise"]):
        return {
           'KMeans': KMeans(n_clusters=3, init='random', algorithm='lloyd'),
            'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
            'DBSCAN': DBSCAN(min_samples=4, metric='euclidean', eps=0.2)
        }

def get_models_for_comparison(dataset: str, algorithm: str):
    if(dataset == "seeds-dataset" and algorithm in ["2d-laplace-truncated", "2d-pairwise"]):
        return {
           '2d-pairwise': KMeans(n_clusters=3, init='random', algorithm='lloyd'),
           '2d-laplace-truncated':KMeans(n_clusters=3, init='random', algorithm='lloyd')
        }
    
def plot_comparison(utility_metrics: pd.DataFrame, dataset, metric = "ami"): 
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='epsilon', y=metric, hue="algorithm", data=utility_metrics, ax=ax)
    ax.set_title(f"Comparison of {metric} for {dataset}. Algorithm: {utility_metrics.iloc[0]['type']}")
    fig.savefig('results/'+metric+'_'+dataset+'_comparison.png')

@app.command()
def run_utility_experiments(plain_dataset_path: str, algorithm: str, dataset: str):
    print(f"Running utility experiments on {plain_dataset_path} with {algorithm}")
    plain_df = helpers.load_dataset(plain_dataset_path)
    print(plain_df.head())
    epsilons = helpers.get_experiment_epsilons()
    sanity_check(algorithm, dataset)
    print('run experiments for: ', epsilons)
    features = dataset_algorithm_features[algorithm][dataset]
    # --- RUN PERTURBATION ALGORITHM ---
    for epsilon in epsilons:    
        Z = get_noise_adding_mechanism(algorithm, plain_df[features], epsilon)
        Z_pd = pd.DataFrame(Z, columns=features)
        Z_pd.to_csv(get_export_path(dataset, algorithm, epsilon), index=False)
        
    supported_models = list(get_models(dataset, algorithm).values())
    print('Generated report for: ', supported_models)
    # --- RUN CLUSTERING ALGORITHMS ---
    report = helpers.generate_external_validity_export(
        epsilons, 
        supported_models,
        import_path=plain_dataset_path, 
        perturbed_path=get_export_path(dataset, algorithm))
    report.to_csv(get_export_path(dataset, algorithm, prefix='results')+'utility_scores.csv', index=False)

    # --- PLOT RESULTS ---
    utility = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm))
    utility.plot_external_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)
    utility_internal = UtilityPlotter.UtilityPlotter(plain_dataset_path, get_models(dataset, algorithm))
    utility_internal.plot_internal_validation(report, get_export_path(dataset, algorithm, prefix='results'), save=True)

@app.command()
def run_comparison_experiment(research_question: str):
    if research_question == 'RQ1':
        print('Running RQ1')
        model_name = helpers.map_models_to_name(get_models(dataset='seeds-dataset', algorithm='2d-laplace-truncated')['KMeans'])
        print('Considering:', model_name)       
        comparison_dp = pd.DataFrame()
        for algorithm in research_question_1_algorithms:
            for dataset in supported_datasets:
                #external_laplace = helpers.load_dataset(get_export_path('seeds-dataset', '2d-laplace-truncated', prefix='results')+'utility_scores.csv')
                ultility_metrics = helpers.load_dataset(get_export_path(dataset, algorithm, prefix='results')+'utility_scores.csv')
                ultility_metrics = ultility_metrics[ultility_metrics['type'] == model_name]
                ultility_metrics['algorithm'] = algorithm
                #print(ultility_metrics.head())
                comparison_dp = pd.concat([comparison_dp, ultility_metrics]).reset_index(drop=True)
        print("Concatenated results:", comparison_dp.head())

        for dataset in supported_datasets:
            plot_comparison(comparison_dp, 'seeds-dataset', metric='ami')
            plot_comparison(comparison_dp, 'seeds-dataset', metric='ari')
            plot_comparison(comparison_dp, 'seeds-dataset', metric='ch')
            plot_comparison(comparison_dp, 'seeds-dataset', metric='sc')
        
@app.command()
def run_privacy_experiments(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")


if __name__ == "__main__":
    app()