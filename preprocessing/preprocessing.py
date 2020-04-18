from sklearn import preprocessing
import numpy as np

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

print("Mean: ", data.mean(axis=0))  # Mean:  [ 1.33333333  1.93333333 -0.06666667 -2.53333333]
print("Standard Deviation: ", data.std(axis=0))  # Standard Deviation:  [1.24721913 2.44449495 1.60069429 3.30689515]

# Standardization

data_standardized = preprocessing.scale(data)

print("Mean standardized data: ",
      data_standardized.mean(axis=0))  # [ 5.55111512e-17 -1.11022302e-16 -7.40148683e-17 -7.40148683e-17]
print("Standard Deviation standardized data: ", data_standardized.std(axis=0))  # [1. 1. 1. 1.]
