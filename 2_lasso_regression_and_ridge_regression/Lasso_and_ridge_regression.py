import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv("week3.csv")
# print(data.head())

data.columns = ['X1', 'X2', 'y']

#Extract Features

X1 = data.iloc[:,0]
X2 = data.iloc[:,1]
X = np.column_stack((X1,X2))
y= data.iloc[:,2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X1, X2, y)

# Set labels for the axes
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# Set a title for the plot
ax.set_title('3D Scatter Plot of Features X1 and X2 vs. Target Variable y')

# Display the plot
plt.show()

degree = 5
penalties = [1, 10, 100]

# Lasso Regression
lasso_results = []
regression_type = 'Lasso'

poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

for penalty in penalties:
    model = Lasso(alpha=1/(2 * penalty))
    model.fit(X_poly, y)
    result_dict = {
        'Penalty (C)': penalty,
        'Intercept': model.intercept_,
        'Coefficients': np.around(model.coef_, decimals=3),
    }
    lasso_results.append(result_dict)

lasso_results_df = pd.DataFrame(lasso_results)
print("Lasso Regression Results:")
print(lasso_results_df)

# Generate a grid of feature values
X_test = []
grid = np.linspace(-5, 5, 100)  # Adjust as needed
for i in grid:
    for j in grid:
        X_test.append([i, j])
X_test = np.array(X_test)

# Polynomial degree and penalty values
degree_poly = 5
c_values = [1, 10, 100]

# Lasso Regression
plot_colors = ['limegreen', 'red']

# Create polynomial features
Xpoly_Ridge = PolynomialFeatures(degree=degree_poly).fit_transform(X)
Xpoly_test = PolynomialFeatures(degree=degree_poly).fit_transform(X_test)

# Loop through penalty values and implement Lasso regression
for c_param in c_values:
    model = Lasso(alpha=1 / (2 * c_param))
    model.fit(Xpoly_Ridge, y)
    predictions = model.predict(Xpoly_test)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predictions
    ax.plot_trisurf(X_test[:, 0], X_test[:, 1], predictions, color=plot_colors[0], alpha=0.5)

    # Plot Training Data
    ax.scatter(X[:, 0], X[:, 1], y, color=plot_colors[1], label='Training Data')

    # Plot configuration
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Lasso Regression, C = {}'.format(c_param), fontdict={'fontsize': 8.5})

    # Legend
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[0], marker='o')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[1], marker='v')
    ax.legend([scatter1_proxy, scatter2_proxy], ['Lasso Predictions', 'Training Data'], numpoints=1)
    ax.view_init(azim=60)

plt.show()

ridge_results = []
regression_type = 'Ridge'

poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

for penalty in penalties:
    model = Ridge(alpha=1/(2 * penalty))
    model.fit(X_poly, y)
    result_dict = {
        'Penalty (C)': penalty,
        'Intercept': model.intercept_,
        'Coefficients': np.around(model.coef_, decimals=3),
    }
    ridge_results.append(result_dict)

ridge_results_df = pd.DataFrame(ridge_results)
print(ridge_results_df)

# Generate a grid of feature values
X_test = []
grid = np.linspace(-5, 5, 100)  # Adjust as needed
for i in grid:
    for j in grid:
        X_test.append([i, j])
X_test = np.array(X_test)

# Polynomial degree and penalty values
degree_poly = 5
c_values = [1, 10, 100]

# Lasso Regression
plot_colors = ['limegreen', 'red']

# Create polynomial features
Xpoly_Ridge = PolynomialFeatures(degree=degree_poly).fit_transform(X)
Xpoly_test = PolynomialFeatures(degree=degree_poly).fit_transform(X_test)

# Loop through penalty values and implement Lasso regression
for c_param in c_values:
    model = Lasso(alpha=1 / (2 * c_param))
    model.fit(Xpoly_Ridge, y)
    predictions = model.predict(Xpoly_test)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predictions
    ax.plot_trisurf(X_test[:, 0], X_test[:, 1], predictions, color=plot_colors[0], alpha=0.5)

    # Plot Training Data
    ax.scatter(X[:, 0], X[:, 1], y, color=plot_colors[1], label='Training Data')

    # Plot configuration
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Lasso Regression, C = {}'.format(c_param), fontdict={'fontsize': 8.5})

    # Legend
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[0], marker='o')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[1], marker='v')
    ax.legend([scatter1_proxy, scatter2_proxy], ['Lasso Predictions', 'Training Data'], numpoints=1)
    ax.view_init(azim=60)

plt.show()

ridge_results = []
regression_type = 'Ridge'

poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

for penalty in penalties:
    model = Ridge(alpha=1/(2 * penalty))
    model.fit(X_poly, y)
    result_dict = {
        'Penalty (C)': penalty,
        'Intercept': model.intercept_,
        'Coefficients': np.around(model.coef_, decimals=3),
    }
    ridge_results.append(result_dict)

ridge_results_df = pd.DataFrame(ridge_results)
print(ridge_results_df)

# Polynomial degree and penalty values
degree_poly = 5
c_values = [1, 10, 100]

# Ridge Regression
# Create polynomial features
Xpoly_Ridge = PolynomialFeatures(degree=degree_poly).fit_transform(X)
Xpoly_test_Ridge = PolynomialFeatures(degree=degree_poly).fit_transform(X_test)

# Loop through penalty values and implement Ridge regression
for c_param in c_values:
    model = Ridge(alpha=1/(2*c_param))
    model.fit(Xpoly_Ridge, y)
    predictions = model.predict(Xpoly_test_Ridge)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predictions
    ax.plot_trisurf(X_test[:, 0], X_test[:, 1], predictions, color=plot_colors[0], alpha=0.5)

    # Plot Training Data
    ax.scatter(X[:, 0], X[:, 1], y, color=plot_colors[1], label='Training Data')

    # Plot configuration
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Ridge Regression, C = {}'.format(c_param), fontdict={'fontsize': 8.5})

    # Legend
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[0], marker='o')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=plot_colors[1], marker='v')
    ax.legend([scatter1_proxy, scatter2_proxy], ['Ridge Predictions', 'Training Data'], numpoints=1)
    ax.view_init(azim=60)

plt.show()

# Define hyperparameters
model_type = 'Lasso'  # Choose 'Lasso' or 'Ridge' as the regression model
C_values = [1, 10, 100]  # Regularization strength values

# Initialize empty lists to store results
mean_error = []
std_error = []
df_results = []

# Generate polynomial features
Xpoly = PolynomialFeatures(degree_poly).fit_transform(X)

# Loop through each C value
for C in C_values:
    mse_train_temp = []
    mse_test_temp = []

    # Create the regression model with the current C value
    if model_type == 'Lasso':
        model = Lasso(alpha=1/(2*C))
    elif model_type == 'Ridge':
        model = Ridge(alpha=1/(2*C))

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5)

    for train, test in kf.split(Xpoly):
        # Fit the model on the training set
        model.fit(Xpoly[train], y[train])

        # Predict on the training set and calculate mean squared error
        ypred_train = model.predict(Xpoly[train])
        mse_train = mean_squared_error(y[train], ypred_train)
        mse_train_temp.append(mse_train)

        # Predict on the test set and calculate mean squared error
        ypred_test = model.predict(Xpoly[test])
        mse_test = mean_squared_error(y[test], ypred_test)
        mse_test_temp.append(mse_test)

    # Create a dictionary of results including mean and variance
    result_dict = {
        'C value': C,
        'mse train results': np.around(mse_train_temp, decimals=3),
        'mean mse train': np.around(np.array(mse_train_temp).mean(), decimals=3),
        'std mse train': np.around(np.array(mse_train_temp).std(), decimals=3),
        'mse test': np.around(mse_test_temp, decimals=3),
        'mean mse test': np.around(np.array(mse_test_temp).mean(), decimals=3),
        'std mse test': np.around(np.array(mse_test_temp).std(), decimals=3)
    }

    df_results.append(result_dict)

# Create a DataFrame to store the results
df_kf_results = pd.DataFrame(df_results)
print(df_kf_results)

# Extract relevant data for plotting
C_values = df_kf_results['C value']
mean_mse_test = df_kf_results['mean mse test']
std_mse_test = df_kf_results['std mse test']

# Create the plot
plt.figure(figsize=(10, 6))
plt.errorbar(C_values, mean_mse_test, yerr=std_mse_test, fmt='-o', color='magenta', label='Mean Test MSE')

# Set plot labels and title
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Mean Square Error')
plt.title('K Fold Cross Validation - Lasso Regression')

# Customize the x-axis scale for better visualization
plt.xscale('log')  # Using a logarithmic scale for C values

# Add a legend
plt.legend()

# Show the plot
plt.show()

# **************** K Fold Cross Validation - Ridge Regression ***************************

# Define hyperparameters
model_type = 'Ridge'  # Choose 'Lasso' or 'Ridge' as the regression model
C_values = [1, 10, 100]  # Regularization strength values

# Initialize empty lists to store results
mean_error = []
std_error = []
df_results = []

# Generate polynomial features
Xpoly = PolynomialFeatures(degree_poly).fit_transform(X)

# Loop through each C value
for C in C_values:
    mse_train_temp = []
    mse_test_temp = []

    # Create the regression model with the current C value
    if model_type == 'Lasso':
        model = Lasso(alpha=1/(2*C))
    elif model_type == 'Ridge':
        model = Ridge(alpha=1/(2*C))

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5)

    for train, test in kf.split(Xpoly):
        # Fit the model on the training set
        model.fit(Xpoly[train], y[train])

        # Predict on the training set and calculate mean squared error
        ypred_train = model.predict(Xpoly[train])
        mse_train = mean_squared_error(y[train], ypred_train)
        mse_train_temp.append(mse_train)

        # Predict on the test set and calculate mean squared error
        ypred_test = model.predict(Xpoly[test])
        mse_test = mean_squared_error(y[test], ypred_test)
        mse_test_temp.append(mse_test)

    # Create a dictionary of results including mean and variance
    result_dict = {
        'C value': C,
        'mse train results': np.around(mse_train_temp, decimals=3),
        'mean mse train': np.around(np.array(mse_train_temp).mean(), decimals=3),
        'std mse train': np.around(np.array(mse_train_temp).std(), decimals=3),
        'mse test': np.around(mse_test_temp, decimals=3),
        'mean mse test': np.around(np.array(mse_test_temp).mean(), decimals=3),
        'std mse test': np.around(np.array(mse_test_temp).std(), decimals=3)
    }

    df_results.append(result_dict)

# Create a DataFrame to store the results
df_kf_results = pd.DataFrame(df_results)
print(df_kf_results)

# Extract relevant data for plotting
C_values = df_kf_results['C value']
mean_mse_test = df_kf_results['mean mse test']
std_mse_test = df_kf_results['std mse test']

# Create the plot
plt.figure(figsize=(10, 6))
plt.errorbar(C_values, mean_mse_test, yerr=std_mse_test, fmt='-o', color='magenta', label='Mean Test MSE')

# Set plot labels and title
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Mean Square Error')
plt.title('K Fold Cross Validation - Ridge Regression')

# Customize the x-axis scale for better visualization
plt.xscale('log')  # Using a logarithmic scale for C values

# Add a legend
plt.legend()

# Show the plot
plt.show()
