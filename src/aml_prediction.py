
"""
Georgios Skoulidis
AML Patient Prediction
"""

#%% Import the Libraries

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier

#%% Load Data

# Load features from data.csv 
features = []
path = 'data/'

with open(path+'data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        features.append([float(val) for val in row])

# Convert the list of features into a NumPy array
features_array = np.array(features)

# Load labels from labels.csv (assuming it's structured similarly)
labels = []
with open(path+'labels.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        labels.append(int(row[0]))  # Assuming the label is in the first column

# Convert the list of labels into a NumPy array
labels_array = np.array(labels)

# Save the array with labeled features
labeled_features_array = features_array[:179]


#%% Metrics Function

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


#%% A Cross-Validation Example to Optimize the Random Forest's Hyperparameters

# Define the estimator grid and hyperparameters
estimator_grid = [5, 10, 25, 40, 60, 100]
hyperparameters = [
    {'class_weight': None, 'max_features': None, 'max_depth': None},
    {'class_weight': {1: 1, 2: 5}, 'max_features': None, 'max_depth': None},
    {'class_weight': {1: 1, 2: 5}, 'max_features': 10, 'max_depth': None},
]

# Store results for comparison
best_model = None
best_val_metric = (0, 0, 0, 0)  # To store best validation metrics (accuracy, precision, recall, f1)
best_params = None

# Cross-validation setup
n_splits = 5
cross_val = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)

# Iterate over the estimator grid and hyperparameters
for n_estimators in estimator_grid:
    for param_set in hyperparameters:
        print(f"Training Random Forest with {n_estimators} estimators and parameters: {param_set}")
        
        # Initialize the Random Forest classifier with current parameters
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            class_weight=param_set['class_weight'], 
            max_features=param_set['max_features'], 
            max_depth=param_set['max_depth'],
            random_state=20
        )

        # Store metrics for each fold
        rf_train_metrics = []
        rf_val_metrics = []

        # Cross-validation for Random Forest with current parameters
        for fold_index, (train_index, val_index) in enumerate(cross_val.split(labeled_features_array, labels_array)):
            print(f"Fold {fold_index + 1}")
            
            # Split the training and validation sets
            x_train, x_val = labeled_features_array[train_index], labeled_features_array[val_index]
            y_train, y_val = labels_array[train_index], labels_array[val_index]

            # Train the Random Forest model
            rf_model.fit(x_train, y_train)
            
            # Make predictions on the training and validation sets
            y_train_pred_rf = rf_model.predict(x_train)
            y_val_pred_rf = rf_model.predict(x_val)

            # Compute metrics for the training and validation sets using the compute_metrics function
            rf_train_metric = compute_metrics(y_train, y_train_pred_rf)
            rf_val_metric = compute_metrics(y_val, y_val_pred_rf)
            
            # Store the computed metrics
            rf_train_metrics.append(rf_train_metric)
            rf_val_metrics.append(rf_val_metric)

        # Compute average metrics across all folds for the current model
        average_rf_train_metrics = np.mean(rf_train_metrics, axis=0)
        average_rf_val_metrics = np.mean(rf_val_metrics, axis=0)

        # Print the average metrics for the current model
        print(f"Average Train Metrics: Accuracy={average_rf_train_metrics[0]:.4f}, Precision={average_rf_train_metrics[1]:.4f}, Recall={average_rf_train_metrics[2]:.4f}, F1={average_rf_train_metrics[3]:.4f}")
        print(f"Average Validation Metrics: Accuracy={average_rf_val_metrics[0]:.4f}, Precision={average_rf_val_metrics[1]:.4f}, Recall={average_rf_val_metrics[2]:.4f}, F1={average_rf_val_metrics[3]:.4f}\n")

        # Check if the current model has better F1 score than the best model so far
        if average_rf_val_metrics[3] > best_val_metric[3]:  # Compare based on F1 score
            best_model = rf_model
            best_val_metric = average_rf_val_metrics
            best_params = {
                'n_estimators': n_estimators,
                'class_weight': param_set['class_weight'],
                'max_features': param_set['max_features'],
                'max_depth': param_set['max_depth']
            }

# Print the best model and its metrics
print("\nBest Random Forest Model:")
print(f"n_estimators: {best_params['n_estimators']}")
print(f"class_weight: {best_params['class_weight']}")
print(f"max_features: {best_params['max_features']}")
print(f"max_depth: {best_params['max_depth']}")
print(f"Best Validation Metrics: Accuracy={best_val_metric[0]:.4f}, Precision={best_val_metric[1]:.4f}, Recall={best_val_metric[2]:.4f}, F1={best_val_metric[3]:.4f}")



#%% Predictions on Unlabeled data with Random Forest 

# Initialize the Random Forest classifier with current parameters
rf_model = RandomForestClassifier(
    n_estimators=25, 
    class_weight={1: 1, 2: 5}, 
    max_features=10, 
    random_state=20
)

# Split the training and validation sets
x_train, x_unlabeled = features_array[:179], features_array[179:]
y_train = labels_array

# Fit model into training data
rf_model.fit(x_train, y_train)
# Predictions for unlabeled data
unlabeled_predictions = rf_model.predict(x_unlabeled)
print("Final label predictions for the unlabeled subset:")
print(unlabeled_predictions)

#%% (Optional): Store Predictions in a csv file

# Create a DataFrame
df = pd.DataFrame(unlabeled_predictions, columns=['Predictions'])

# Save DataFrame to CSV
output_file_path = 'Final_prediction.csv'  # Specify your desired output file name
df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}.")

