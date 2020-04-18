# In supervised learning, we usually deal with a variety of labels. These can be either numbers or words. If they are
# numbers, then the algorithm can use them directly. However, labels often need to be in a human-readable form. So,
# people usually label the training data with words.

# Label encoding refers to transforming word labels into a numerical form so that algorithms can understand how to
# operate on them

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print("Class mapping: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)
# audi --> 0
# bmw --> 1
# ford --> 2
# toyota --> 3

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("Labels =", labels)
print("Encoded labels =", list(encoded_labels))  # [3, 2, 0]

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("Encoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_labels))  # ['ford', 'bmw', 'audi', 'toyota', 'bmw']

# NB: Label encoding can transform categorical data into numeric data, but the imposed ordinality creates problems if
# the obtained values are submitted to mathematical operations.One-hot encoding has the advantage that the result is
# binary rather than ordinal, and that everything is in an orthogonal vector space. The disadvantage is that for high
# cardinality, the feature space can explode.
