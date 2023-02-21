import zipfile
import pandas as pd


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
