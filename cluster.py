from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_dataPath = r'C:/Users/conta/source/repos/PythonClassifierIris/PythonClassifierIris/Data/iris.data'

def import_data():

    import os

    assert os.path.isfile(_dataPath)
    with open(_dataPath, 'r') as f:
        pass

    data = open(_dataPath, 'r')

    from pandas import read_fwf

    frame = read_fwf(data)

    frame = read_table(
        _dataPath,      
        encoding='utf-8',
        sep=',',
        skipinitialspace=True,
        index_col=None,
        header=None,
    )

    return frame


def get_features_and_labels(frame):

    arr = np.array(frame)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    enc = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], 
                            remainder='passthrough')

    arr = np.array(enc.fit_transform(arr))

    X, y = arr, arr

    print(X)
    print(y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_classifier(X_train, X_test, y_train, y_test):

    from sklearn.cluster import KMeans
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import MeanShift
    from sklearn.cluster import SpectralClustering

    from sklearn.metrics import silhouette_score, average_precision_score

    print('X_train: ', X_train, 'X_test: ', X_test, 'y_train', y_train, 'y_test', y_test)
    
    cluster = KMeans(n_clusters=2, random_state=0)
    cluster.fit(X_train, y_train)

    score = silhouette_score(y_test, cluster.predict(X_test))
    
    print(score)

    yield 'KMeans (Silhoutte_score={:.3f})'.format(score)

    cluster = AffinityPropagation(preference=-50, random_state=5)
    cluster.fit(X_train, y_train)

    score = silhouette_score(y_test, cluster.predict(X_test))

    print('\n', score)

    yield 'AffinityPropagation (Silhoutte_score={:.3f})'.format(score)

    cluster = SpectralClustering(n_clusters=2)

    cluster.fit_predict(X_train, y_train)
    score = silhouette_score(y_test, cluster.fit_predict(X_test))

    print('\n', score)

    yield 'SpectralClustering (Silhoutte_score={:.3f})'.format(score)


def plot(results):

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + _dataPath)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':

    frame = import_data()

    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    #print("Plotting the results")
    #plot(results)
