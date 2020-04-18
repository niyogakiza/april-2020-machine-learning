### A regressor can be evaluated using many different metrics, such as the following:

- Mean absolute error: This is the average of absolute errors of all the data points in the given dataset.
- Mean squared error: This is the average of the squares of the errors of all the data points in the given dataset. It is one of the most popular metrics out there!
- Median absolute error: This is the median of all the errors in the given dataset. The main advantage of this metric is that it's robust to outliers. A single bad point in the test dataset wouldn't skew the entire error metric, as opposed to a mean error metric.
- Explained variance score: This score measures how well our model can account for the variation in our dataset. A score of 1.0 indicates that our model is perfect.
- R2 score: This is pronounced as R-squared, and this score refers to the coefficient of determination. This tells us how well the unknown samples will be predicted by our model. The best possible score is 1.0, but the score can be negative as well.


- Functions ending with  _score return a value to maximize; the higher the better
- Functions ending with _error or _loss return a value to minimize; the lower the better 
