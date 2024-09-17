# Import necessary libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Assume X is your feature matrix and y is your target vector
# Load your data accordingly
# Example:
# X, y = load_your_data_function()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.1 Solution of Ridge Regression and Lasso
# Set regularization parameter λ = 0.1 and compute the number of nonzero coefficients
lambda_value = 0.1

# Ridge Regression
ridge = Ridge(alpha=lambda_value)
ridge.fit(X_train, y_train)
ridge_w = ridge.coef_

# Lasso Regression
lasso = Lasso(alpha=lambda_value / len(y_train))  # lambda/n for Lasso in sklearn
lasso.fit(X_train, y_train)
lasso_w = lasso.coef_

# Number of nonzero coefficients
ridge_nonzero = np.sum(ridge_w != 0)
lasso_nonzero = np.sum(lasso_w != 0)

print(f'Nonzero coefficients in Ridge: {ridge_nonzero}')
print(f'Nonzero coefficients in Lasso: {lasso_nonzero}')

#----------------------------------------------------------------------

# 3.2 Training and Testing Error with Different Values of λ
lambdas = [0, 1e-5, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]

ridge_errors_train = []
ridge_errors_test = []
lasso_errors_train = []
lasso_errors_test = []

ridge_nonzeros = []
lasso_nonzeros = []

ridge_norms = []
lasso_norms = []

#3.2.1 
for lmbda in lambdas:
    # Ridge regression
    ridge = Ridge(alpha=lmbda)
    ridge.fit(X_train, y_train)

    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge.predict(X_train)))
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
    ridge_errors_train.append(ridge_train_rmse)
    ridge_errors_test.append(ridge_test_rmse)

    ridge_nonzeros.append(np.sum(ridge.coef_ != 0))

    ridge_norms.append(np.linalg.norm(ridge.coef_, 2))

    # Lasso regression
    lasso = Lasso(alpha=lmbda / len(y_train))
    lasso.fit(X_train, y_train)

    lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso.predict(X_train)))
    lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    lasso_errors_train.append(lasso_train_rmse)
    lasso_errors_test.append(lasso_test_rmse)

    lasso_nonzeros.append(np.sum(lasso.coef_ != 0))
    lasso_norms.append(np.linalg.norm(lasso.coef_, 2))

#3.2.2 Plot RMSE vs Lambda
plt.figure(figsize=(10, 6))
plt.plot(lambdas, ridge_errors_train, label='Ridge Train RMSE')
plt.plot(lambdas, ridge_errors_test, label='Ridge Test RMSE')
plt.plot(lambdas, lasso_errors_train, label='Lasso Train RMSE')
plt.plot(lambdas, lasso_errors_test, label='Lasso Test RMSE')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend()
plt.title('RMSE vs Lambda')
plt.show()


# 3.2.3 Number of Nonzero Coefficients vs λ
plt.figure(figsize=(10, 6))
plt.plot(lambdas, ridge_nonzeros, label='Ridge Nonzero Coefficients')
plt.plot(lambdas, lasso_nonzeros, label='Lasso Nonzero Coefficients')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Number of Nonzero Coefficients')
plt.legend()
plt.title('Nonzero Coefficients vs Lambda')
plt.show()

# 3.2.4 ||w||_2 vs λ
plt.figure(figsize=(10, 6))
plt.plot(lambdas, ridge_norms, label='Ridge ||w||_2')
plt.plot(lambdas, lasso_norms, label='Lasso ||w||_2')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('||w||_2')
plt.legend()
plt.title('||w||_2 vs Lambda')
plt.show()

#----------------------------------------------------------------------
# 3.3 Cross-Validation
ridge_scores = []
lasso_scores = []
for lmbda in lambdas:
    # Ridge Cross-Validation
    ridge = Ridge(alpha=lmbda)
    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(np.mean(np.sqrt(-ridge_cv_scores)))
    
    # Lasso Cross-Validation
    lasso = Lasso(alpha=lmbda / len(y_train))
    lasso_cv_scores = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(np.mean(np.sqrt(-lasso_cv_scores)))

print(f'Best Ridge λ: {lambdas[np.argmin(ridge_scores)]}, Cross-Validation RMSE: {min(ridge_scores)}')
print(f'Best Lasso λ: {lambdas[np.argmin(lasso_scores)]}, Cross-Validation RMSE: {min(lasso_scores)}')