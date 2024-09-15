import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_files
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Load the E2006-tfidf dataset
X_train, y_train, X_test, y_test = load_svmlight_files(('E2006.train', 'E2006.test'))  # Adjust path as needed
num_samples = X_train.shape[0] #should be 16087
# print(num_samples)
                                                        
# Initialize Ridge and Lasso models
# You may want to increase max_iter for Lasso model to guarantee convergence
lamda = 0.1
### YOUR CODE HERE

### YOUR CODE HERE

# Train the Ridge and Lasso model
### YOUR CODE HERE

### YOUR CODE HERE

# Make prediction with both models
ridge_predictions_train = ridge.predict(X_train)
lasso_predictions_train = lasso.predict(X_train)
ridge_predictions = ridge.predict(X_test)
lasso_predictions = lasso.predict(X_test)

# Evaluate models using Root Mean Squared Error
# To compute RMSE, you can use the function "mean_squared_error" by setting 'squared=False'
### YOUR CODE HERE

### YOUR CODE HERE

# Print result and Plot curve
### YOUR CODE HERE
# Note: You can use matplotlib to plot curve


### YOUR CODE HERE

# Perform 5-fold cross-validation procedure
### YOUR CODE HERE




### YOUR CODE HERE