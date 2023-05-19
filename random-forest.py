import numpy as np
import os
from pathlib import Path
import image_pre_processing_functions as ipf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

PATH_PROJECT = Path(__file__).parent
PATH_GEDI = Path(__file__).parent / "GEDI_resampled"
DIRECTORIES = ['B1_subset', 'B2_subset', 'B3_subset', 'B4_subset', 'B5_subset', 'B6_subset', 'B7_subset', 'B8_subset',
               'B9_subset', 'B10_subset', 'B11_subset', 'B12_subset', 'evi_subset', 'ndvi_subset', 'ndmi_subset', 'ndre_subset']
TEST_DIRECTORIES = ['B1_subset']
all_features = []

for dir in os.listdir(PATH_PROJECT / 'B1_subset'):


# for directory in TEST_DIRECTORIES:
#     feature = os.path.join(PATH_PROJECT, directory)
#     concatenated_images_one_feature = ipf.concat_all_images(feature)
#     print (feature, concatenated_images_one_feature.shape)

    # flattened_concatenated_images_one_feature = ipf.flatten_variable(concatenated_images_one_feature)
    # all_features.append(flattened_concatenated_images_one_feature)
    # print(all_features)

# One input array for all images, normalised with min-max and subsetted, all clean:)
# multiple features for each sample. hence axis=1
# combined_data = np.concatenate(all_features, axis=0)


# # Prepare the target variable (canopy height) as a 1D numpy array
# concatenated_gedi_resampled = ipf.concat_all_images(PATH_GEDI)
# canopy_height = ipf.flatten_variable(concatenated_gedi_resampled)
# print(canopy_height.shape)

# Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(combined_data, canopy_height, test_size=0.2, random_state=42)

# # Train the random forest model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Evaluate the model on the validation set
# predictions = rf_model.predict(X_val)
#
# # Assess model performance (you can use any evaluation metric suitable for regression)
# mse = np.mean((predictions - y_val) ** 2)
# rmse = np.sqrt(mse)
# r2 = rf_model.score(X_val, y_val)