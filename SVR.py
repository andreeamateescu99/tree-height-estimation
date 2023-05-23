import numpy as np
import os
from pathlib import Path
import utility_functions as uf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_PROJECT = Path(__file__).parent
PATH_GEDI = Path(__file__).parent / "GEDI_resampled"
DIRECTORIES = ['B1_subset', 'B2_subset', 'B3_subset', 'B4_subset', 'B5_subset', 'B6_subset', 'B7_subset', 'B8_subset',
               'B9_subset', 'B10_subset', 'B11_subset', 'B12_subset', 'evi_subset', 'ndvi_subset', 'ndmi_subset', 'ndre_subset']

TEST_DIRECTORIES = ['B1_subset', ]

# Prepare input
X = uf.prepare_feature_matrix(PATH_PROJECT, DIRECTORIES)

# # Prepare the target variable (canopy height) as a 1D numpy array
target_variable = uf.stack_one_feature('/Users/andreeamateescu/PycharmProjects/tree-height-estimation/GEDI_resampled')
y = np.nan_to_num(target_variable, nan=0.0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost regression model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Evaluate the model on the testing dataset
y_pred = svr_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

# # Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

uf.plot_predictions(y_val, y_pred, 'svr')
