import zipfile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, roc_curve, silhouette_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.utils import to_categorical
from art.attacks.inference.membership_inference import ShadowModels
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from Helpers import twod_laplace
from diffprivlib.mechanisms import laplace, gaussian
from scipy.spatial import KDTree
from itertools import cycle
from Helpers.pairwise import PMBase, PiecewiseMechanism
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

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

def load_plain_and_perturbed_dataset(epsilon, import_path, perturbed_path):
    dataset_name1 = import_path
    dataset_name2 = perturbed_path + 'perturbed_' + str(epsilon) + '.csv'
    dataset1 = load_dataset(dataset_name1)
    dataset2 = load_dataset(dataset_name2)
    return dataset1, dataset2

def get_experiment_epsilons():
    epsilons = [0.01, 0.03, 0.05, 0.07, 0.1 , 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 5, 7, 9]
    return epsilons
    
def map_models_to_name(model): 
    parameters = model.get_params()
    model_name = type(model).__name__
    if(model_name == 'KMeans'):
        return 'KMeans(clusters='+str(parameters['n_clusters'])+', init='+parameters['init']+')'
    elif(model_name == 'DBSCAN'):
        return 'DBSCAN(samples='+str(parameters['min_samples'])+', distance_metric='+parameters['metric']+', eps='+str(parameters['eps'])+')'
    elif(model_name == 'AffinityPropagation'):
        return 'AffinityPropagation(damping='+str(parameters['damping'])+', distance_metric='+parameters['affinity']+')'
    elif(model_name == 'OPTICS'):
        return 'OPTICS(min_samples='+str(parameters['min_samples'])+', distance_metric='+parameters['metric']+')'
    else: 
        return 'Not supported'
    

def measure_external_validity_report(epsilon, cluster_model, import_path, perturbed_path, columns):
    plain_df, perturbed_df = load_plain_and_perturbed_dataset(epsilon, import_path, perturbed_path)
    plain_df = plain_df[columns]
    perturbed_df = perturbed_df[columns]
    plain_df_scaled = StandardScaler().fit_transform(plain_df)
    perturbed_df_scaled = StandardScaler().fit_transform(perturbed_df)
    plain_fitted_df = cluster_model.fit(plain_df_scaled)
    perturbed_fitted_df = clone(cluster_model).fit(perturbed_df_scaled)
    ami = adjusted_mutual_info_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
    ari = adjusted_rand_score(plain_fitted_df.labels_, perturbed_fitted_df.labels_)
    ch = calinski_harabasz_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
    sc = silhouette_score(perturbed_df_scaled, perturbed_fitted_df.labels_)
    return ami, ari, ch, sc
    
def generate_external_validity_export(epsilons, models, n_times = 10, import_path='../exports', perturbed_path='../perturbed', model_name= None, columns=['X', 'Y']):
    dataframe = {'type': [], 'epsilon': [], 'ari': [], 'ami': [], 'ch': [], 'sc': []}
    for epsilon in epsilons:
        for model in models:
            algorithmName = model_name if model_name is not None else map_models_to_name(model)
            dataframe['type'].append(algorithmName)
            dataframe['epsilon'].append(epsilon)
            ami_list = []
            ari_list = []
            ch_list = []
            sc_list = []
            for i in range(n_times):
                ami, ari, ch, sc = measure_external_validity_report(epsilon, model, import_path, perturbed_path, columns)
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

    
def run_mi_experiments(X, y_true, epsilons, n_times = 10, algorithm = None, targets = 4, columns=['X', 'Y']): 
    shokri_mi_avgs = {'epsilon': [], 'shokri_mi_adv': [], 'attack_adv': [], 'tpr': [], 'fpr': [],'run': []}
    X_pd = pd.DataFrame(X, columns=columns)
    #create_labels = KMeans(init='random', n_clusters=4)
    #create_labels.fit(StandardScaler().fit_transform(X_pd))
    #X_pd['target'] = create_labels.labels_
    for epsilon in epsilons:
        #_, _, Z = twod_laplace.generate_truncated_laplace_noise(X, epsilon)
        #Z_pd = pd.DataFrame(Z, columns=['X', 'Y'])
        #create_labels = KMeans(init='random', n_clusters=4)
        #create_labels.fit(StandardScaler().fit_transform(Z_pd))
        #target = create_labels.labels_
        for run in range(n_times):
            shokri_mi_avgs['epsilon'].append(epsilon)
            shokri_mi_avgs['run'].append(run)

            shadow_ratio = 0.75
            dataset = train_test_split(X_pd, y_true, test_size=shadow_ratio)

            x_target, x_shadow, y_target, y_shadow = dataset

            attack_train_size = len(x_target) // 2
            #attack_test_size = attack_train_size
            x_target_train = algorithm(x_target[:attack_train_size], epsilon)
            x_target_train = np.array(x_target_train)
            #x_target_train = X_pd.iloc[x_target[:target_train_size].index, 0:2]
            y_target_train = y_target[:attack_train_size]
            #y_target_train = X_pd.iloc[x_target[:target_train_size].index, 2]
            x_target_test = x_target[attack_train_size:]
            y_target_test = y_target[attack_train_size:]

            # We infer based on the original data, to make sure we can estimate the dp protection
            #x_shadow_np = X_pd.iloc[x_shadow.index, 0:2].to_numpy()
            #y_shadow_np = X_pd.iloc[y_shadow.index, 2].to_numpy()
            x_shadow_np = np.array(x_shadow)
            y_shadow_np = y_shadow
            clf = RandomForestClassifier()
            classifier = clf.fit(x_target_train, y_target_train)
            
            art_classifier = ScikitlearnRandomForestClassifier(classifier)

            ## train shadow models
            shadow_models = ShadowModels(art_classifier, num_shadow_models=3)
            shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow_np, to_categorical(y_shadow_np, targets))
            (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

            ## Execute membership attack
            attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
            attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)

            member_infer = attack.infer(x_target_train, y_target_train)
            nonmember_infer = attack.infer(x_target_test, y_target_test)

            # concatenate everything and calculate roc curve
            predicted_y = np.concatenate((member_infer, nonmember_infer))
            actual_y = np.concatenate((np.ones(len(member_infer)), np.zeros(len(nonmember_infer))))
            fpr, tpr, _ = roc_curve(actual_y, predicted_y, pos_label=1)
            attack_adv = tpr[1] / (tpr[1] + fpr[1])
            print(tpr[1], fpr[1])
            shokri_mi_avgs['shokri_mi_adv'].append(tpr[1] - fpr[1])
            shokri_mi_avgs['attack_adv'].append(attack_adv)
            shokri_mi_avgs['tpr'].append(tpr)
            shokri_mi_avgs['fpr'].append(fpr)

    return pd.DataFrame(shokri_mi_avgs)

def generate_piecewise_perturbation(plain_df, epsilon):
    plain_df = reshape_data_to_uniform(plain_df) # resphape to [-1, 1] (rquired by the algorithm)
    pm_encoder = PiecewiseMechanism(epsilon=epsilon, domain=[-1.001, 1.001])
    perturbed_df = plain_df.copy()
    for col in plain_df.columns:
        perturbed_df[col] = plain_df[col].apply(pm_encoder.randomise)
    return perturbed_df

def kDistancePlot(X):
    neigh = NearestNeighbors(n_neighbors=2)
    neighbours = neigh.fit(X)
    distances, indices = neighbours.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.title('K-distance plot for estimating the epsilon DBSCAN')
    plt.xlabel('Data points')
    plt.ylabel('Epsilon')
    plt.plot(distances)

def reshape_data_to_uniform(dataframe: pd.DataFrame):
    return pd.DataFrame(MinMaxScaler(feature_range=(-1, 1)).fit_transform(dataframe.values), columns=dataframe.columns)


def truncate_n_dimensional_laplace_noise(perturbed_df: np.array, plain_df: np.array, grid_size=10):
    mesh = [np.linspace(plain_df[:, i].min(), plain_df[:, i].max(), num=grid_size) for i in range(plain_df.shape[1])]
    meshgrid = np.meshgrid(*mesh, indexing='ij')
     # Create a KDTree from dataset2
    tree = KDTree(plain_df)

    # Query the KDTree with dataset1 to find the closest points in dataset2
    _, closest_indices = tree.query(perturbed_df)

    # Calculate the distances between dataset1 and closest points in dataset2
    distances = np.linalg.norm(perturbed_df - plain_df[closest_indices], axis=1)

    # Reshape the meshgrid array
    meshgrid_reshaped = np.stack(meshgrid, axis=-1)

    # Create a KDTree from meshgrid
    meshgrid_tree = KDTree(meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1]))

    # Query the KDTree with dataset1 to find the closest points in meshgrid
    _, closest_meshgrid_indices = meshgrid_tree.query(perturbed_df)

    # Calculate the distances between dataset1 and closest points in meshgrid
    meshgrid_distances = np.linalg.norm(perturbed_df - meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1])[closest_meshgrid_indices], axis=1)

    # Check if each point in dataset1 is within the domain of dataset2
    in_domain = np.logical_and.reduce([np.logical_and(perturbed_df[:, dim] >= plain_df[:, dim].min(), perturbed_df[:, dim] <= plain_df[:, dim].max()) for dim in range(perturbed_df.shape[1])])

    # Create a mask for points outside the domain of dataset2
    outside_domain_mask = np.logical_not(in_domain)

    # Create a mask for points outside the domain and closer to meshgrid points
    outside_domain_and_closer_mask = np.logical_and(outside_domain_mask, meshgrid_distances < distances)

    # Remap points outside the domain and closer to meshgrid points to the closest meshgrid points
    remapped_dataset = perturbed_df.copy()
    remapped_dataset[outside_domain_and_closer_mask] = meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1])[closest_meshgrid_indices][outside_domain_and_closer_mask]

    return remapped_dataset


def prepare_for_roc(mi_scores):
    mi_scores_for_display = mi_scores.copy()
    #mi_scores_eps_1 = mi_scores_for_display[mi_scores_for_display['epsilon'] == 0.1]
    mi_scores_for_display['tpr'] = mi_scores_for_display['tpr'].apply(lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x)
    mi_scores_for_display['fpr'] = mi_scores_for_display['fpr'].apply(lambda x: float(x.strip('[]').split()[1]) if type(x) is not float else x)
    extra_point = pd.DataFrame({'fpr': [0.0], 'tpr': [0.0]})
    #extra_point2 = pd.DataFrame({'fpr': [1.0], 'tpr': [1.0]})
    return pd.concat([mi_scores_for_display, extra_point], ignore_index=True)

def display_roc_plot(mi_scores, types, title = 'ROC Curve', save_as = None):
    #fig, ax = plt.subplots(figsize=(8, 6))
    #display = RocCurveDisplay(fpr=mi_scores_for_display['fpr'].sort_values(), tpr=mi_scores_for_display['tpr'].sort_values()).plot(ax)
    line_styles = cycle(['-', '--', '-.', ':'])
    fig, ax = plt.subplots(figsize=(12, 10))

    # sns.lineplot(x=mi_scores['fpr'].sort_values(), y=mi_scores['tpr'].sort_values())
    for score_type in types:
        linestyle = next(line_styles)
        mi_scores_for_display = prepare_for_roc(mi_scores.loc[mi_scores['algorithm'] == score_type])
        plt.plot(mi_scores_for_display['fpr'].sort_values(), mi_scores_for_display['tpr'].sort_values(), lw=2, label=score_type, linestyle=linestyle)
    #plt.plot(mi_scores['fpr'].sort_values(), mi_scores['tpr'].sort_values(), lw=2)
    # disabling the offset on y axis

    ax.set_title(title)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("Flase Positive Rate")
    ax.plot([0, 1], [0, 1],'r--', label='Random Guess')
    ax.set_xlim([-0.05, 1.02])  # Set the x-axis limits from 0 to 1
    ax.set_ylim([-0.05, 1.02])  # Set the y-axis limits from 0 to 1
    ax.legend(loc="lower right")
    if(save_as is not None):
        fig.savefig(save_as)