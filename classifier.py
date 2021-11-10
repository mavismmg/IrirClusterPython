from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_dataPath = r'C:/Users/conta/source/repos/PythonClassifierIris/PythonClassifierIris/Data/iris.data'

try:
    import seaborn
except ImportError:
    pass


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

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(arr[:, :-1]).toarray())

    arr = arr.append(enc_df, how='left')

    print(frame)

    X, y = arr[:, :-1], arr[:, -1]

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

    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    classifier = KMeans(n_clusters=2, random_state=0)

    classifier.fit(X_train, y_train)
    score = silhouette_score(y_test, classifier.predict(X_test))

    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = accuracy_score(y_test, y_prob)

    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)

    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))

    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall


    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')

    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))

    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall


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

    print("Plotting the results")
    plot(results)
