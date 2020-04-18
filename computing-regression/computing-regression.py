import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
import numpy as np

# Regression is used to find out the relationship between input data and the continuously-valued output data. This is
# generally represented as real numbers, and our aim is to estimate the core function that calculates the mapping
# from the input to the output.

# 1 --> 2
# 3 --> 6
# 4.3 --> 8.6
# 7.1 --> 14.2
# this matches this function: f(x) = 2(x)

filename = "VehiclesItaly.txt"

x = []
y = []

with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

# When we build a machine learning model, we need a way to validate our model and check whether it is performing at a
# satisfactory level. To do this, we need to separate our data into two groups—a training dataset and a testing
# dataset. The training dataset will be used to build the model, and the testing dataset will be used to see how this
# trained model performs on unknown data. So, let's go ahead and split this data into training and testing datasets:

num_training = int(0.8 * len(x))
num_test = len(x) - num_training

# Training data

x_train = np.array(x[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])

# Test data

x_test = np.array(x[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])

# Create linear regressor object

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets

linear_regressor.fit(x_train, y_train)
y_train_pred = linear_regressor.predict(x_train)

# To plot training outputs, we will use the matplotlib

plt.figure()
plt.scatter(x_train, y_train, color='orange')
plt.plot(x_train, y_train_pred, color='blue', linewidth=4)
plt.title('Training data')
plt.show()

# To plot test output

y_test_pred = linear_regressor.predict(x_test)
plt.figure()
plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

# compute regression accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print('================================')
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print('================================')
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print('================================')
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print('================================')
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Mean absolute error = 241907.27
# ================================
# Mean squared error = 81974851872.13
# ================================
# Median absolute error = 240861.94
# ================================
# Explain variance score = 0.98
# ================================
# R2 score = 0.98

# NB: An R2 score near 1 means that the model is able to predict the data very well. Keeping track of every single
# metric can get tedious, so we pick one or two metrics to evaluate our model. A good practice is to make sure that
# the mean squared error is low and the explained variance score is high.
