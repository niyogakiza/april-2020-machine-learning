# Binarization Binarization is used when you want to convert a numerical feature vector into a Boolean vector. In the
# field of digital image processing, image binarization is the process by which a color or grayscale image is
# transformed into a binary image, that is, an image with only two colors (typically, black and white).
# This technique is used for the recognition of objects, shapes, and, specifically, characters. Through binarization,
# it is possible to distinguish the object of interest from the background on which it is found. Skeletonization is
# instead an essential and schematic representation of the object, which generally preludes the subsequent real
# recognition.

from sklearn import preprocessing
import numpy as np

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print(data_binarized)
# [[1. 0. 1. 0.]
#  [0. 1. 0. 1.]
#  [0. 1. 0. 0.]]

# NB: Binarization is a widespread operation on count data, in which the analyst can decide to consider only the
# presence or absence of a characteristic rather than a quantified number of occurrences. Otherwise, it can be used
# as a preprocessing step for estimators that consider random Boolean variables.

