# Normalization                                                        Data normalization is used when you want to
# adjust the values in the feature vector so that they can be measured on a common scale. One of the most common
# forms of normalization that is used in machine learning adjusts the values of a feature vector so that they sum up
# to 1.
from sklearn import preprocessing
import numpy as np

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

data_normalized = preprocessing.normalize(data, norm='l1', axis=0)
print(data_normalized)

# [[ 0.75       -0.17045455  0.47619048 -0.45762712]
#  [ 0.          0.45454545 -0.07142857  0.1779661 ]
#  [ 0.25        0.375      -0.45238095 -0.36440678]]

data_norm_abs = np.abs(data_normalized)
print(data_norm_abs.sum(axis=0))
# [1. 1. 1. 1.] the sum of the absolute value of the elements of each column is
# equal to 1, so the data is normalized
