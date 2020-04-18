from sklearn import preprocessing
import numpy as np

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

data_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1))

data_scaled = data_scaled.fit_transform(data)
print("Min: ", data.min(axis=0))  # [ 0.  -1.5 -1.9 -5.4]
print("Min: ", data.max(axis=0))  # [3.  4.  2.  2.1]

# display the scaled array
print(data_scaled)
# [[1.         0.         1.         0.        ]
#  [0.         1.         0.41025641 1.        ]
#  [0.33333333 0.87272727 0.         0.14666667]]
