# import numpy
# import scipy
# import matplotlib
# import sklearn

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
# (506, 13)

data, target = load_boston(return_X_y=True)
print(data.shape)
# (506, 13)
print(target.shape)
# (506,)
