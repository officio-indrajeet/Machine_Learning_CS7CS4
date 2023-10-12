# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ************************************************ Part A : Logistic Regression Model *******************************
# Part 1: Data Loading and Visualization

# Load the dataset from a CSV file
data = pd.read_csv('week2.csv')
data.columns = ['X1', 'X2', 'y']

# Scatter plot of the data
def plot_scatter(data):
    plt.scatter(data.loc[data['y'] == 1, 'X1'], data.loc[data['y'] == 1, 'X2'], marker='+', c='lime', alpha=0.5,
                label='Training data, y = 1')
    plt.scatter(data.loc[data['y'] == -1, 'X1'], data.loc[data['y'] == -1, 'X2'], marker='o', c='blue',
                label='Training data, y = -1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Scatter Plot of Features X1 and X2')
    plt.legend(['Training data, y = 1', 'Training data, y = -1'], fancybox=True, framealpha=1, bbox_to_anchor=(1.04, 1),
               loc="upper left")
    plt.show()

plot_scatter(data)

# Part 2: Logistic Regression Model
# Separate features and target
X = data[['X1', 'X2']]
y = data['y']

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the model to the data
logistic_model.fit(X, y)

# Parameter Values
intercept = logistic_model.intercept_
coef = logistic_model.coef_

# Make predictions
predictions = logistic_model.predict(X)
data['predict'] = predictions

accuracy = accuracy_score(data['y'], data['predict'])*100
# print(accuracy)

confusion_mat = confusion_matrix(data['y'],data['predict'])
# print(confusion_mat)

# Decision boundary calculation
decision_boundary = (-coef[0, 0] / coef[0, 1]) * data['X1'] - intercept[0] / coef[0, 1]

# Visualize the logistic regression model with custom markers and colors
def plot_logistic_regression(data, coef, intercept):
    # Scatter plot of the model's predictions
    plt.scatter(data.loc[data['predict'] == 1, 'X1'], data.loc[data['predict'] == 1, 'X2'], marker='o',
                facecolors='none', edgecolors='purple', label='Predictions, y = 1')
    plt.scatter(data.loc[data['predict'] == -1, 'X1'], data.loc[data['predict'] == -1, 'X2'], marker='o',
                facecolors='none', edgecolors='orange', label='Predictions, y = -1')

    # Scatter plot of the training data
    plt.scatter(data.loc[data['y'] == 1, 'X1'], data.loc[data['y'] == 1, 'X2'], marker='+', c='lime', alpha=0.7,
                label='Training data, y = 1')
    plt.scatter(data.loc[data['y'] == -1, 'X1'], data.loc[data['y'] == -1, 'X2'], marker='+', c='blue', alpha=0.7,
                label='Training data, y = -1')

    # Plot Decision Boundary
    plt.plot(data['X1'], decision_boundary, linewidth=5, c='red', label='Decision boundary')

    # Labels, Title, and Legend
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logistic Regression Model Visualization')
    plt.legend(fancybox=True, framealpha=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

# Visualize the logistic regression model
plot_logistic_regression(data, coef, intercept)

# ****************************************** Part B: Support Vector Machine ******************************************
# Read data from 'week2.csv' and store it in a DataFrame 'df1'
df1 = pd.read_csv('week2.csv')
# Rename the columns of the DataFrame to 'X1', 'X2', and 'y'
df1.columns = ['X1', 'X2', 'y']

# Extract features 'X1' and 'X2' into 'svm_X'
svm_X = data[['X1', 'X2']]
# Extract target 'y' into 'svm_y'
svm_y = data['y']

# Create LinearSVC model
def train_linear_svm(svm_X, svm_y, c_param):
    svc_model = LinearSVC(C=c_param, dual=True, max_iter=10000)
    svc_model.fit(svm_X, svm_y)
    return svc_model

# Range of C values
c_values_to_test = [0.001, 1, 100]

# Train LinearSVC models for a range of C values and collect results
def train_and_collect_results(svm_X, svm_y, c_values):
    svc_results = []
    for c_param in c_values:
        svc_model = train_linear_svm(svm_X, svm_y, c_param)
        model_results = {
            'C_value': c_param,
            'intercept': svc_model.intercept_[0],
            'coef_0': svc_model.coef_[0, 0],
            'coef_1': svc_model.coef_[0, 1],
        }
        svc_results.append(model_results)
    return pd.DataFrame(svc_results)

# Implement & get dataframe
df_svc_results = train_and_collect_results(svm_X, svm_y, c_values_to_test)

# Part (ii) Plot Data, Predictions & Decision Boundary
# Function to visualize the impact of different C parameters on Linear Support Vector Classification (LinearSVC)
def visualize_linear_svm_range(data, c_test, plot_dim):
    # Create a figure for plotting
    fig = plt.figure(figsize=(15, 10))
    count = 0
    # Extract features and target labels from the dataset
    X1 = df1['X1']
    X2 = df1['X2']
    svm_X = np.column_stack((X1, X2))
    svm_y = df1['y']

    # Loop through different C parameters and train LinearSVC models
    for c_param in c_test:
        count += 1
        # Train a LinearSVC model with the current C parameter
        svc_model = train_linear_svm(svm_X, svm_y, c_param)

        # Make predictions using the trained model
        predictions = svc_model.predict(svm_X)
        df1['predict'] = predictions

        # Create a subplot for the current C parameter
        plt.subplot(plot_dim[0], plot_dim[1], count)

        # Plot predicted points for class '1' as purple circles
        plt.scatter(df1.loc[df1['predict'] == 1, 'X1'], df1.loc[df1['predict'] == 1, 'X2'], marker='o',
                    facecolors='none', edgecolors='purple', label='Predictions, y = 1')
        # Plot predicted points for class '-1' as orange circles
        plt.scatter(df1.loc[df1['predict'] == -1, 'X1'], df1.loc[df1['predict'] == -1, 'X2'], marker='o',
                    facecolors='none', edgecolors='orange', label='Predictions, y = -1')
        # Plot actual points for class '1' as green plus signs
        plt.scatter(df1.loc[df1['y'] == 1, 'X1'], df1.loc[df1['y'] == 1, 'X2'], marker='+', c='lime', alpha=0.7,
                    label='Training data, y = 1')
        # Plot actual points for class '-1' as blue plus signs
        plt.scatter(df1.loc[df1['y'] == -1, 'X1'], df1.loc[df1['y'] == -1, 'X2'], marker='+', c='blue', alpha=0.7,
                    label='Training data, y = -1')

        # Decision boundary
        decision_boundary = (-coef[0, 0] / coef[0, 1]) * X1 - intercept[0] / coef[0, 1]
        plt.plot(df1['X1'], decision_boundary, linewidth=5, c='red', label='Decision boundary')

        # Labels
        plt.title('Linear Support Vector Classification (LinearSVC)\nExploring Different C Parameters: C = %.3f' % c_param)
        plt.xlabel('X1')
        plt.ylabel('X2')

    plt.legend(fancybox=True, framealpha=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()



# Range of C values
c_test = [0.001, 1, 100]
# Plot dimensions
plot_dim = [2, 2]

# Implement with a range of C values
visualize_linear_svm_range(data, c_test, plot_dim)

accuracy_for_svm_model = accuracy_score(df1['y'],df1['predict'])*100
# print(accuracy_for_svm_model)

confusion_mat_for_svm_model = confusion_matrix(df1['y'],df1['predict'])
# print(accuracy_for_svm_model)

# Focus on two specific values of C
c_test = [0.001, 100]
# Plot dimensions
plot_dim = [1, 2]

# Implement with specific C values
visualize_linear_svm_range(data, c_test, plot_dim)

# ************************************ Part C : Logistic Regression on additional features **********************

# Create a copy of the dataset
df = data.copy()

# Part (i): Data Preprocessing and Logistic Regression Model Training
# Create additional features by squaring X1 and X2
df['squared_X1'] = df['X1'] ** 2
df['squared_X2'] = df['X2'] ** 2

# Define features and target
squared_X1 = df['squared_X1']
squared_X2 = df['squared_X2']
X_with_squared_features = np.column_stack((X, squared_X1, squared_X2))
y = df['y']

# Train a logistic regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_with_squared_features, y)

# Get model coefficients
intercept = log_reg_model.intercept_
coefficients = log_reg_model.coef_

# Part (ii): Predictions and Plotting

# Make predictions using the trained model
predictions = log_reg_model.predict(X_with_squared_features)

# Create a column for predictions in the DataFrame
df['predict'] = predictions

accuracy_for_log_regression_with_additional_features = accuracy_score(df['y'], df['predict']) * 100
# print(accuracy_for_log_regression_with_additional_features)


def plot_logistic_regression_results(df, log_reg_model):
    # Plot of the model's predictions
    plt.scatter(df.loc[df['predict'] == 1, 'X1'], df.loc[df['predict'] == 1, 'X2'], marker='o',
                facecolors='none', edgecolors='purple', label='Predictions, y = 1')
    plt.scatter(df.loc[df['predict'] == -1, 'X1'], df.loc[df['predict'] == -1, 'X2'], marker='o',
                facecolors='none', edgecolors='orange', label='Predictions, y = -1')
    # Plot of Training Data
    plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker='+', c='lime', alpha=0.7,
                label='Training data, y = 1')
    plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker='+', c='blue', alpha=0.7,
                label='Training data, y = -1')
    # Labels, Title, and Legend
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logistic Regression Model Visualization with Additional Features')
    plt.legend(fancybox=True, framealpha=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

plot_logistic_regression_results(df,log_reg_model)

# Part (iii): Baseline Model
# Check class distribution of the target variable

class_distribution = df['y'].value_counts()
# print(class_distribution)

df['predict_col_1'] = np.ones(len(y))

# Visualizes the results of a baseline model and actual training data using the 'X1' and 'X2' features.
def plot_logistic_regression_baseline_results(df, log_reg_model):
    # Plot Baseline Predictions
    plt.scatter(df.loc[df['predict_col_1'] == 1, 'X1'], df.loc[df['predict_col_1'] == 1, 'X2'], marker='o',
                facecolors='none', edgecolors='purple', label='Baseline Predictions, y = 1')
    plt.scatter(df.loc[df['predict_col_1'] == -1, 'X1'], df.loc[df['predict_col_1'] == -1, 'X2'], marker='o',
                facecolors='none', edgecolors='orange', label='Baseline Predictions, y = -1')

    # Plot of Training Data
    plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker='+', c='lime', alpha=0.7,
                label='Training data, y = 1')
    plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker='+', c='blue', alpha=0.7,
                label='Training data, y = -1')

    # Labels, Title, and Legend
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Baseline Model')
    plt.legend(fancybox=True, framealpha=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

plot_logistic_regression_baseline_results(df,log_reg_model)





