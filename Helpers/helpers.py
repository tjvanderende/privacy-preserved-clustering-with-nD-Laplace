import zipfile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, silhouette_score
import seaborn as sns

def elbow_plot(sse):
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


def calculate_kmeans_sum_square_errors(data):
    from sklearn.cluster import KMeans
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_
    return sse


def import_dasaset(name, **kwargs):
    import pandas as pd
    archive = zipfile.ZipFile(name, 'r')

    if ("filename" in kwargs):
        value = archive.open(kwargs.get('filename'))
        return pd.read_csv(value, sep='\\t')
    else:
        training = archive.open('train.csv')
        test = archive.open('test.csv')
        training_pd = pd.read_csv(training)
        test_pd = pd.read_csv(test)
        return [training_pd, test_pd]


def remove_missing(dataframe: pd.DataFrame):
    print('Rows without missing values: %n', dataframe.size)
    without_missing = dataframe.dropna()
    print('Rows with missing values: %n', without_missing.size)
    return without_missing


def load_dataset(datasetname):
    df = pd.read_csv(datasetname)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def load_plain_and_perturbed_dataset(epsilon, import_path):
    dataset_name1 = import_path + 'plain.csv'
    dataset_name2 = import_path + 'perturbed_' + str(epsilon) + '.csv'
    dataset1 = load_dataset(dataset_name1)
    dataset2 = load_dataset(dataset_name2)
    return dataset1, dataset2

def get_experiment_epsilons():
    epsilons = [0.05, 0.1 , 0.5 , 1, 2, 3, 5, 7, 9]
    return epsilons
    
def map_models_to_name(model): 
    parameters = model.get_params()
    model_name = type(model).__name__
    if(model_name == 'KMeans'):
        return 'KMeans(clusters='+str(parameters['n_clusters'])+', init='+parameters['init']+')'
    elif(model_name == 'DBSCAN'):
        return 'DBSCAN(samples='+str(parameters['min_samples'])+', distance_metric='+parameters['metric']+', epsilon='+str(parameters['eps'])+')'
    elif(model_name == 'AffinityPropagation'):
        return 'AffinityPropagation(damping='+str(parameters['damping'])+', distance_metric='+parameters['affinity']+')'
    else: 
        return 'Not supported'
    

def measure_external_validity_report(epsilon, cluster_model, import_path):
    plain_df, perturbed_df = load_plain_and_perturbed_dataset(epsilon, import_path)
    plain_df_scaled = StandardScaler().fit_transform(plain_df)
    perturbed_df_scaled = StandardScaler().fit_transform(perturbed_df)
    plain_fitted_df = cluster_model.fit(plain_df_scaled)
    perturbed_fitted_df = clone(cluster_model).fit(perturbed_df_scaled)
    ami = adjusted_mutual_info_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
    ari = adjusted_rand_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
    ch = calinski_harabasz_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
    sc = silhouette_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
    return ami, ari, ch, sc
    
def generate_external_validity_export(epsilons, models, n_times = 10, import_path='../exports'):
    dataframe = {'type': [], 'epsilon': [], 'ari': [], 'ami': [], 'ch': [], 'sc': []}
    for epsilon in epsilons:
        for model in models:
            algorithmName = map_models_to_name(model)
            dataframe['type'].append(algorithmName)
            dataframe['epsilon'].append(epsilon)
            ami_list = []
            ari_list = []
            ch_list = []
            sc_list = []
            for i in range(n_times):
                ami, ari, ch, sc = measure_external_validity_report(epsilon, model, import_path)
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


def plot_utility(dataframe, epsilons, metric_name, axes, metric = 'Adjusted Mutual Information (AMI)', title='External validation (AMI & ARI): Difference between privately trained cluster algorithms versus \n non-private trained cluster algorithms'): 
    ax = sns.lineplot(x='epsilon', y=metric_name, data=dataframe, ax=axes, style='type', hue='type', markers=True, legend=True)
    ax.set_xticks(epsilons, labels=epsilons)
    ax.set_title(title)
    ax.set_xlabel('Privacy budget ($\epsilon$)')
    ax.set_ylabel(metric)
    plt.legend(title='Cluster algorithm', loc='upper left', labels=dataframe['type'].unique())

