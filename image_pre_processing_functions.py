import os

import numpy as np

from utility_functions import read_GeoTiff


def normalize_array(arr: np.ndarray) -> np.ndarray:
    X_min = np.nanmin(arr)
    X_max = np.nanmax(arr)
    X_mean = np.nanmean(arr)
    range_nonzero = (X_max - X_min) != 0
    X_scaled = np.where(range_nonzero, (arr - X_mean) / (X_max - X_min), np.nan)
    return X_scaled


def concat_all_images(feature_folder: str) -> np.ndarray:
    arrays = []
    for image in os.listdir(feature_folder):
        if image == '.DS_Store':
            continue
        array, profile = read_GeoTiff(os.path.join(feature_folder, image))
        arrays.append(normalize_array(array))  # dopamine
    # multiple samples with the same features but from different time periods, axis = 0
    concatenated_array = np.concatenate(arrays, axis=1)

    return concatenated_array


def flatten_variable(array: np.ndarray) -> np.ndarray:
    return array.flatten()


def target_variable_assertions(original_array, flattened_array):
    expected_flattened_shape = (original_array.shape[0] * original_array.shape[1],)
    assert flattened_array.shape == expected_flattened_shape, "Flattened array has incorrect shape."

# x= concat_split_all_images('/Users/andreeamateescu/PycharmProjects/tree-height-estimation/B6_subset')
# print(type(y))
# print(z[~np.isnan(z)])
