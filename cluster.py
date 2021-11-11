import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas import read_table

_dataPath = r'C:/Users/conta/source/repos/PythonClassifierIris/PythonClassifierIris/Data/iris.data'

def import_data():
    # Verify if file in path is not a folder, if folder then pass
    assert os.path.isfile(_dataPath)
    with open(_dataPath, 'r') as f:
        pass

    # Loading data from _dataPath as read -> open(path, 'as what?')
    data = open(_dataPath, 'r')

    from pandas import read_fwf
    # Using pandas for reading data as .data format -> read_fwf(data) and creating a frame for the dataset
    frame = read_fwf(data)
    # Defining parameters for table read_table(path, separator, skipinitialspace=bool, index_col=value, header=value)
    frame = read_table(
        _dataPath,      
        sep=',',
        skipinitialspace=True,
        index_col=None,
        header=None,
    )

    return frame


def get_features_and_labels(frame):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    # Defining and creating a numpy array and loading data from frame
    arr = np.array(frame)
    # Defining encoder variable as enc, and then applying a encoding method
    enc = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], 
                            remainder='passthrough')

    # Adding the encoded data in the numpy array
    arr = np.array(enc.fit_transform(arr))
    # Selecting all columns minus the first three
    X, y = arr[:, 3:], arr[:, 3:]

    from sklearn.model_selection import train_test_split
    # Split arrays or matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    from sklearn.preprocessing import StandardScaler
    # Loading StandardScaler method
    scaler = StandardScaler()
    # Standardize features by removing the mean and scaling to unit variance
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

    yield 'KMeans (Silhoutte_score={:.3f})'.format(score), score

    cluster = AffinityPropagation(preference=-50, random_state=5)
    cluster.fit(X_train, y_train)

    score = silhouette_score(y_test, cluster.predict(X_test))

    print('\n', score)

    yield 'AffinityPropagation (Silhoutte_score={:.3f})'.format(score), score

    cluster = SpectralClustering(n_clusters=2)

    cluster.fit_predict(X_train, y_train)
    score = silhouette_score(y_test, cluster.fit_predict(X_test))

    print('\n', score)

    yield 'SpectralClustering (Silhoutte_score={:.3f})'.format(score), score


def cluster_visualizing():
    frame = import_data()
    # Pre-processing frame data
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    arr = np.array(frame)
    enc = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], 
                            remainder='passthrough')

    arr = np.array(enc.fit_transform(arr))
    X, y = arr[:, 3:], arr[:, 3:]
    # Loading variables from frame
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)
    from sklearn.decomposition import PCA
    # Create a PCA model to reduce our data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    pca.fit(X_test)
    # Transfor the scaled data to the new PCA space
    X_reduced = pca.transform(X_test)
    X_reducedDf = pd.DataFrame(X_test, columns=['Column1', 'Column2'])
    X_reducedDf['Cluster'] = clusters
    X_reducedDf.head()


def plot(results):

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + _dataPath)

    for label, score in results:
        plt.plot(score, label=label)

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

    #cluster_visualizing()

    print("Plotting the results")
    plot(results)
