from matplotlib import pyplot as plt
from Helpers import UtilityPlotter
import Helpers.helpers as help
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans


def models(): 
    return {
        'KMeans': KMeans(n_clusters=4, init='random', algorithm='lloyd'),
        'AffinityPropagation': AffinityPropagation(damping=0.5, affinity='euclidean'),
        'DBSCAN': DBSCAN(min_samples=6, eps=0.6, metric='euclidean')
    }

# import export
utility_metrics = help.load_dataset('./export/results/utility-3d.csv')
utility_metrics.head()

# run experiments
plotter = UtilityPlotter.UtilityPlotter('./export/plain.csv', models())
plotter.plot_external_validation(utility_metrics, export_path='./export/results/', save=True)

