#coding:utf-8

from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

kRow, kCol = 2, 3
kComponents = kRow * kCol


def plot_gallery(title, images, n_col, n_row):
    image_shape = (64, 64)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())  # 去除坐标轴
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)


if __name__ == '__main__':
    data_set = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
    faces = data_set.data

    plot_gallery('First centered Olivetti faces', faces[:kComponents], kCol, kRow)

    # estimators = [
    #     ('Eigenfaces - PCA using randomized SVD', decomposition.PCA(n_components=6, whiten=True)),
    #     ('Non-negative components - NMF', decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
    # ]
    #
    #
    # for name, estimator in estimators:
    #     print("Extracting the top %d %s..." % (n_components, name))
    #     print(faces.shape)
    #     estimator.fit(faces)
    #     components_ = estimator.components_
    #     plot_gallery(name, components_[:n_components])

    name = 'Eigenfaces - PCA using randomized SVD'
    estimator = decomposition.PCA(n_components=6, whiten=True)
    print("Extracting the top %d %s..." % (kComponents, name))
    print(faces.shape)
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name, components_[:kComponents], kCol, kRow)

    # name = 'Non-negative components - NMF'
    # estimator = decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3)
    # print("Extracting the top %d %s..." % (n_components, name))
    # print(faces.shape)
    # estimator.fit(faces)
    # components_ = estimator.components_
    # plot_gallery(name, components_[:n_components], kCol, kRow)

    plt.show()
