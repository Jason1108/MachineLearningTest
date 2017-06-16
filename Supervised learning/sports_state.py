import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))

    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label


if __name__ == '__main__':
    featurePaths = ['data/A.feature','data/B.feature','data/C.feature','data/D.feature','data/E.feature']
    labelPaths = ['data/A.label','data/B.label','data/C.label','data/D.label','data/E.label']

    x_train, y_train = load_datasets(featurePaths[:4], labelPaths[:4])
    x_test, y_test = load_datasets(featurePaths[4:], labelPaths[4:])

    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)

    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')

    print('The classification report for knn:')
    print(classification_report(y_test, answer_knn))