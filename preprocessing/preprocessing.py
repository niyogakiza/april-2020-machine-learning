# In the real world, we usually have to deal with a lot of raw data. This raw data is not readily ingestible by
# machine learning algorithms. To prepare data for machine learning, we have to preprocess it before we feed it into
# various algorithms. This is an intensive process that takes plenty of time, almost 80 percent of the entire data
# analysis process, in some scenarios. However, it is vital for the rest of the data analysis workflow,
# so it is necessary to learn the best practices of these techniques. Before sending our data to any machine learning
# algorithm, we need to cross check the quality and accuracy of the data. If we are unable to reach the data stored
# in Python correctly, or if we can't switch from raw data to something that can be analyzed, we cannot go
# ahead. Data can be preprocessed in many ways—standardization, scaling, normalization, binarization, and one-hot
# encoding are some examples of preprocessing techniques. We will address them through simple examples.

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
