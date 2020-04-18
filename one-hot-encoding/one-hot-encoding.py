# One hot encoding
# We often deal with numerical values that are sparse and scattered all over the place. We don't
# really need to store these values. This is where one-hot encoding comes into the picture. We can think of one-hot
# encoding as a tool that tightens feature vectors. It looks at each feature and identifies the total number of
# distinct values. It uses a one-of-k scheme to encode values. Each feature in the feature vector is encoded based on
# this scheme. This helps us to be more efficient in terms of space.

from sklearn import preprocessing
import numpy as np

data = np.array([[1, 1, 2], [0, 2, 3], [1, 0, 1], [0, 1, 0]])
print(data)

encoder = preprocessing.OneHotEncoder()
encoder.fit(data)

encoded_vector = encoder.transform([[1, 2, 3]]).toarray()

print(encoded_vector)
# [[0. 1. 0. 0. 1. 0. 0. 0. 1.]] The result is clear: the first feature (1) has an index of 1, the second feature (3)
# has an index of 4, and the third feature (3) has an index of 8. As we can verify, only these positions are occupied
# by a 1; all the other positions have a 0. Remember that Python indexes the positions starting from 0,
# so the 9 entries will have indexes from 0 to 8.


