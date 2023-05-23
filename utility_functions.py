import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma
import plotly.graph_objects as go
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.profiles import Profile
from rasterio.warp import calculate_default_transform, reproject


# GEOTIFF RELATED
def read_GeoTiff(file: str) -> Tuple[np.ndarray, Profile]:
    with rio.open(file) as src:
        return src.read(), src.profile


def resample_GeoTiff(input_path: str, dst_bounds: Tuple[float, float, float, float], dst_shape: Tuple[int, int],
                     dst_crs: CRS = None,
                     resampling_method: Resampling = Resampling.nearest, output_path: Optional[str] = None) -> Tuple[
    np.ndarray, Profile]:
    """
        Resamples a GeoTIFF file located at input_path to a new resolution, extent, and CRS specified by dst_bounds, dst_shape,
        and dst_crs parameters, respectively. The resampling is performed using the specified resampling_method.

        Args:
            input_path (str): Path to the input GeoTIFF file.
            dst_bounds (Tuple[float, float, float, float]): Tuple representing the bounding box of the desired output extent
                in the order (left, bottom, right, top).
            dst_shape (Tuple[int, int]): Tuple representing the desired output shape in pixels (height, width).
            dst_crs (CRS, optional): Desired output CRS. If not provided, the CRS of the input GeoTIFF is used.
            resampling_method (Resampling, optional): Resampling method to be used. Defaults to Resampling.nearest.
            output_path (str, optional): Path to save the resampled GeoTIFF file. If not provided, the function returns the
                resampled data array and the profile without saving the file.

        Returns:
            Tuple[np.ndarray, Profile]: If output_path is None, returns a tuple containing the resampled data array as a
                NumPy array and the profile as a rasterio Profile object. If output_path is provided, returns None, None.
    """
    dst_height, dst_width = dst_shape

    with rio.open(input_path) as src:
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else np.nan

        if dst_crs:
            dst_crs = dst_crs.to_string()

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, dst_width, dst_height, *dst_bounds
        )

        vrt_options = {
            'resampling': resampling_method,
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            'crs': dst_crs,
            'nodata': src_nodata,
            'dtype': src.dtypes[0]  # Use the same data type as the source
        }

        if output_path is not None:
            profile = src.profile.copy()
            profile.update(vrt_options)
            profile.pop('driver', None)

            with rio.open(output_path, 'w', **profile) as dst:
                reproject(
                    source=src.read(1),
                    destination=rio.band(dst, 1),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method,
                    dst_nodata=src_nodata
                )

            return None, None  # Return None values as the operation is write-only

        else:
            resampled = np.zeros((src.count, dst_height, dst_width), dtype=src.dtypes[0])
            reproject(
                source=src.read(indexes=1),
                destination=resampled,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling_method,
                src_nodata=src_nodata,
                dst_nodata=np.nan
            )

            profile = src.profile.copy()

            # Update the transform and CRS in the profile
            profile.update({
                'transform': dst_transform,
                'height': dst_height,
                'width': dst_width,
                'crs': dst_crs,
            })

            # Create a masked array to handle nodata values
            masked_resampled = ma.masked_equal(resampled, np.nan)

            return masked_resampled, profile


def check_geotiff_files(input_file, output_file):
    """
    Checks the consistency of two GeoTIFF files by comparing their extent, dimensions, transformation matrices,
    spatial reference systems (SRS), array shapes, and resolutions.

    Args:
        input_file (str): Path to the first GeoTIFF file.
        output_file (str): Path to the second GeoTIFF file.

    Returns:
        None

    Raises:
        AssertionError: If any of the checked attributes between the two GeoTIFF files do not match.
    """
    with rio.open(input_file) as src1, rio.open(output_file) as src2:
        try:
            assert src1.bounds == src2.bounds, f"Extent of the geotiff files do not match: {src1.bounds}, {src2.bounds}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.width == src2.width, f"Number of pixels in the x direction do not match: {src1.width}, {src2.width}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.height == src2.height, f"Number of pixels in the y direction do not match: {src1.height}, {src2.height}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.transform == src2.transform, f"Transformation matrices do not match: {src1.transform}, {src2.transform}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.crs == src2.crs, f"Spatial reference system (SRS) and projection do not match: {src1.crs}, {src2.crs}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.read().shape == src2.read().shape, f"Shapes of the arrays do not match: {src1.read().shape}, {src2.read().shape}"
        except AssertionError as e:
            print(str(e))
        try:
            assert src1.res == src2.res, f"Resolutions do not match: {src1.res}, {src2.res}"
        except AssertionError as e:
            print(str(e))


def write_GeoTiff(file_path: str, data: np.ndarray, profile: Profile):
    """
    Writes the provided NumPy array data as a GeoTIFF file at the specified file_path using the provided profile.

    Args:
        file_path (str): Path to the output GeoTIFF file to be created.
        data (np.ndarray): NumPy array containing the data to be written as the raster values.
        profile (Profile): Profile object containing metadata information for the GeoTIFF file.

    Returns:
        None
    """
    with rio.open(file_path, 'w', **profile) as dst:
        dst.write(data)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    X_min = np.nanmin(arr)
    X_max = np.nanmax(arr)
    X_mean = np.nanmean(arr)
    range_nonzero = (X_max - X_min) != 0
    X_scaled = np.where(range_nonzero, (arr - X_mean) / (X_max - X_min), np.nan)
    return X_scaled


def filter_corrupted_null(folder_path: str) -> None:
    """
    Check and remove corrupted or null GeoTIFF files in the given folder.

    Args:
        folder_path (str): Path to the folder containing GeoTIFF files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            file_path = os.path.join(folder_path, filename)
            try:
                with rio.open(file_path) as src:
                    arr = src.read(1)
                    if len(arr[~np.isnan(arr)]) == 0 or sum(arr[~np.isnan(arr)]) == 0:
                        os.remove(file_path)
                        print(f"Removed corrupted file: {filename}")
            except Exception as e:
                print(f"Error processing file: {filename}, {e}")


def stack_one_feature(folder_path):
    """
       Stacks raster images from a folder to create a feature matrix.

       Args:
           folder_path (str): Path to the folder containing raster images in TIFF format.

       Returns:
           numpy.ndarray: One-dimensional feature matrix representing pixel values across time.
    """
    stacked_bands = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            file_path = os.path.join(folder_path, filename)
            with rio.open(file_path) as src:
                band = src.read(1, masked=True)  # consider nan as no data
                band = band.filled(src.nodata)
                stacked_bands.append(normalize_array(band))
    # stack all pictures from one folder
    stacked_bands = np.stack(stacked_bands, axis=0)  # stack them on top of each other to represent time
    # the atribute 0 of the stacked bands must equal the number of images or months of retrieved data
    num_timeframes, height, width = stacked_bands.shape
    # get the number of pixels
    num_pixels = height * width
    # the first attribute of the final feature matrix must be the number of pixels in each band and they must correspond
    # to the number of pixels in all the GEDI samples
    reshaped_stacked_feature = stacked_bands.reshape(num_pixels, num_timeframes)
    # flatten across time
    one_feature_matrix = reshaped_stacked_feature.flatten()
    return one_feature_matrix


def prepare_feature_matrix(parent_path: Path, list_features: List) -> np.ndarray:  # ['B1', 'B2']
    """
        Prepares a feature matrix by stacking multiple feature matrices together.

        Args:
            parent_path (str): Path to the parent folder containing individual feature folders.
            list_features (list): List of feature folder names to be included in the feature matrix.

        Returns:
            numpy.ndarray: Two-dimensional feature matrix where each column represents a different feature.
    """
    feature_paths = []
    for feat in list_features:
        feature_paths.append(os.path.join(parent_path, feat))
    feature_matrix = []
    for feature in feature_paths:
        feature_matrix.append(stack_one_feature(feature))
    X = np.stack(feature_matrix, axis=1)
    return X


# PLOT
def plot_ground_truth_vs_predicted(y_true, y_pred):
    """
    Plot ground truth values against predicted values.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
    """
    # Create a scatter plot
    fig = go.Figure()

    # Add a scatter trace for ground truth vs. predicted values
    fig.add_trace(go.Scatter(
        x=y_true,  # Ground truth values
        y=y_pred,  # Predicted values
        mode='markers',
        name='Ground Truth vs. Predicted',
        marker=dict(
            color='blue',
            symbol='circle'
        )
    ))

    # Update layout
    fig.update_layout(
        title='Ground Truth vs. Predicted Values',
        xaxis=dict(title='Ground Truth'),
        yaxis=dict(title='Predicted'),
    )
    with open(f"ground_truth_vs_predictions.html", "a") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    # Show the plot

def plot_predictions(y_test, y_pred, model):
    # Create a scatter plot of actual vs. predicted values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Tree Height')
    plt.ylabel('Predicted Tree Height')
    plt.title('Actual vs. Predicted Tree Height')
    plt.savefig(f'pred {model}.png')
    plt.show()


# PIPELINE RELATED
def create_directories(parent_directory, directory_names):
    for name in directory_names:
        directory_path = os.path.join(parent_directory, name)
        os.makedirs(directory_path, exist_ok=True)


def only_common_dates(list1, list2):
    """
        Filter file paths based on common dates in their filenames.

        Args:
            list1 (List[str]): List of file paths.
            list2 (List[str]): Another list of file paths.
    """
    # Extract the dates from each list
    dates1 = [item.split("/")[-1:][0].split("_")[0] for item in list1]
    dates2 = [item.split("/")[-1:][0].split("_")[0] for item in list2]

    # Find the common dates
    common_dates = set(dates1) & set(dates2)

    # Filter the lists based on common dates
    filtered_list1 = [item for item in list1 if item.split("/")[-1:][0].split("_")[0] in common_dates]
    filtered_list2 = [item for item in list2 if item.split("/")[-1:][0].split("_")[0] in common_dates]

    return filtered_list1, filtered_list2


def get_abs_paths(directory):
    """
        Retrieve absolute paths of files within a directory.

        Args:
            directory (str or Path): Path to the directory containing the files.

        Returns:
            List[str]: A list of absolute paths to the files in the directory, sorted based on the dates in their names.
    """
    absolute_paths = []

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Get the absolute path of each file
        if filename == '.DS_Store':
            continue
        file_path = os.path.abspath(os.path.join(directory, filename))

        # Add the absolute path to the list
        absolute_paths.append(file_path)

    # Sort the file paths based on the dates in their names
    absolute_paths.sort(
        key=lambda x: datetime.strptime(x.split("/")[-1:][0].split("_")[0], "%Y-%m-%d"))

    return absolute_paths
