from Helpers import helpers
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

def run_security_mi_for_dimensions_and_algorithm(X: pd.DataFrame, algorithm_names, epsilons, n_times=1, target_column='class'):
    column_size = X.shape[1]
    x_without_target = X.copy()
    x_without_target.drop(columns=[target_column], inplace=True)
    y_true = X[target_column]
    targets = len(X[target_column].unique())
    total_df = pd.DataFrame()
    for algorithm_name in algorithm_names:
        algorithm = helpers.get_mechanism(algorithm_name)
        for col in range(4, column_size + 1):
            print('Adding one column each time...')
            data = x_without_target.iloc[:,0:col]
            columns = data.columns
            print(f"data-shape: {data.shape}")
            security_df = helpers.run_mi_experiments(data, y_true, epsilons, n_times=n_times, columns=columns, algorithm=algorithm, targets=targets)
            security_df['dimensions'] = col
            security_df['mechanism'] = algorithm_name
            total_df = pd.concat([total_df, security_df], ignore_index=False)

    return total_df

def plot_mi_heatmap(mi_scores_df, dataset, save_path=None):
    fig, ax = plt.subplots(figsize=(15, 8))

    prepared_df = mi_scores_df.copy()
    prepared_df['tpr'] = prepared_df['tpr'].apply(
        lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x)
    prepared_df_mean = prepared_df.groupby(['epsilon', 'dimensions'])['tpr'].mean().reset_index()
    prepared_df_pivot = prepared_df_mean.pivot(index='epsilon', columns='dimensions', values='tpr')
    sns.heatmap(prepared_df_pivot, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap="Blues")
    ax.set_title(f"MI Scores for dataset: {dataset} with epsilon and dimensions")
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Privacy budgets($\epsilon$)')
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def run_for_dimensions_and_algorithms(X: pd.DataFrame, epsilon, model, perturbing_mechanisms, n_times = 10, dataset=None, import_path='../data/heart-dataset/heart_numerical.csv', perturbed_path=''):
    if(dataset is None):
        raise Exception('Dataset cannot be None')
    
    column_size = X.shape[1]
    dataframe = pd.DataFrame();
    for algorithm in perturbing_mechanisms:
        print('Running for algorithm ' + algorithm)
        perturbed_path_cp = perturbed_path + algorithm + '/' + dataset + '/'
        data = run_for_dimensions(X, epsilon, column_size, model, n_times, algorithm, import_path, perturbed_path_cp)
        dataframe = pd.concat([dataframe, data], ignore_index=True)
    return dataframe

def run_for_dimensions(perturbed_df, epsilon, max_columns, models, n_times = 10, model_name = None, import_path='./', perturbed_path='../ExperimentRunners/data/nd-laplace-truncated/heart-dataset/'):
    dataframe = {'type': [], 'dimensions': [], 'ari': [], 'ami': [], 'ch': [], 'sc': []}

    for col in range(4, max_columns+1):
        print('Adding one column each time...')
        data = perturbed_df.iloc[:,0:col]
        columns = data.columns
        print(data.shape)

        for model in models:
            algorithm_name = model_name if model_name is not None else helpers.map_models_to_name(model)
            dataframe['type'].append(algorithm_name)
            dataframe['dimensions'].append(col)
            ami_list = []
            ari_list = []
            ch_list = []
            sc_list = []
            for i in range(n_times):
                ami, ari, ch, sc = helpers.measure_external_validity_report(epsilon, model, import_path=import_path, perturbed_path=perturbed_path, columns=columns)
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

def run_for_dimensions_and_epsilons(X: pd.DataFrame, models, perturbing_mechanisms, epsilons, n_times = 10, dataset=None, import_path='../data/heart-dataset/heart_numerical.csv', save_path=None):
    complete_set = pd.DataFrame()
    for epsilon in epsilons:
        print('Running for epsilon ' + str(epsilon))
        dataframe = run_for_dimensions_and_algorithms(X, epsilon, models, perturbing_mechanisms, n_times, import_path=import_path, dataset=dataset)
        dataframe['epsilon'] = epsilon
        complete_set = pd.concat([complete_set, dataframe], ignore_index=True)

    if save_path is not None:
        complete_set.to_csv(save_path + '.csv')
    return complete_set


def plot_dimensions(metric: str, epsilon, dataframe, dataset, ylabel='Adjusted Rand Index (ARI)', xlabel='dimensions', save_path=None):
    dataframe_for_epsilon = dataframe.copy()
    dataframe_for_epsilon = dataframe_for_epsilon[dataframe_for_epsilon['epsilon'] == epsilon]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title(f"Scores per dimension, with metric: {metric.upper()} and epsilon: {epsilon} for dataset: {dataset}")
    #ax.set_title('Esilon ' + str(epsilon))
    ax.set_xlabel(xlabel)    
    ax.set_ylabel(ylabel)
    sns.lineplot(x='dimensions', y=metric, data=dataframe_for_epsilon, style='type', hue='type', errorbar=None, markers=True, legend=True, ax=ax)
    #plt.title(f'{metric} for different dimensions and epsilons'.capitalize())

    if save_path is not None:
        fig.savefig(save_path + '.png')
        plt.clf()
    else:
        plt.show()


def plot_mi_dimensions(epsilon, dataframe, dataset, ylabel='Mutual Information (MI)', xlabel='dimensions', save_path=None):
    dataframe_for_epsilon = dataframe.copy()
    dataframe_for_epsilon = dataframe_for_epsilon[dataframe_for_epsilon['epsilon'] == epsilon]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title(f"Scores per dimension, for mechanisms: nd-laplace-optimal-truncated and piecewise with {epsilon} for dataset: {dataset}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    twinx = ax.twinx()
    twinx.set_ylabel('True Positive Rate (TPR)')
    dataframe_for_epsilon = helpers.prepare_for_roc(dataframe_for_epsilon)
    sns.lineplot(x='dimensions', y='shokri_mi_adv', data=dataframe_for_epsilon, hue='mechanism', style='mechanism', errorbar=None, markers=True, legend=True, ax=ax)
    sns.lineplot(x='dimensions', y='tpr', data=dataframe_for_epsilon, hue='mechanism', style='mechanism', errorbar=None, markers=True, legend=True, ax=twinx, alpha=0.3)
    plt.legend(['nd-laplace-optimal-truncated', 'piecewise', 'nd-laplace-optimal-truncated (TPR)', 'piecewise (TPR)'])
    if save_path is not None:
        fig.savefig(save_path + '.png')
        plt.clf()
    else:
        plt.show()