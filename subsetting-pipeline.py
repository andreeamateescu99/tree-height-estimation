import os
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling

import utility_functions as uf

PATH_PROJECT = Path(__file__).parent
PATH_GEDI = Path(__file__).parent / "GEDI"
INPUT_FOLDER_RESOLUTION = 'B1'
INPUT_FILE_RESOLUTION_PATH = PATH_PROJECT / INPUT_FOLDER_RESOLUTION / '2019-04-01_2019-05-01_S2-B1_vosges-mountains-subset.tif'
GEDI_RESAMPLED = Path(__file__).parent / "GEDI_resampled"
DIRECTORY_NAMES = ['B2_subset', 'B3_subset', 'B4_subset', 'B5_subset', 'B6_subset', 'B7_subset', 'B8_subset',
                   'B9_subset', 'B10_subset', 'B11_subset', 'B12_subset', 'evi_subset', 'ndre_subset', 'ndvi_subset',
                   'ndmi_subset']  # , 'B1', 'B2',


# uf.create_directories(PATH_PROJECT, DIRECTORY_NAMES)

# function that changes resolution to all GEDI images based on one image that is unprocessed and contains all info
def gedi_resolution(path_gedi: Path, input_file: Path, output_folder: Path):
    """
        Resamples GeoTIFF files in the given folder (path_gedi) to match the resolution and extent of the input_file.
        The resampled files are saved in the output_folder.

        Args:
            path_gedi (Path): Path to the folder containing GeoTIFF files to be resampled.
            input_file (Path): Path to the input GeoTIFF file that defines the desired resolution and extent.
            output_folder (str): Path to the folder where the resampled files will be saved.

        Returns: None
    """
    with rio.open(input_file) as src_input:
        crs = rio.CRS.from_epsg(4326)
        shape = (src_input.height, src_input.width)
        bounds = src_input.bounds
    resampling_method = Resampling.nearest
    for filename in os.listdir(path_gedi):
        if filename == '.DS_Store':
            continue
        file_path = os.path.join(path_gedi, filename)
        if os.path.isfile(file_path):
            gedi_small, gedi_profile = uf.resample_GeoTiff(file_path, bounds, shape, crs, resampling_method)
            to_rename = file_path.split("/")[6].rsplit(".")[0] + "_resampled.tif"
            output_file_path = output_folder / to_rename
            os.chdir(output_folder)
            uf.write_GeoTiff(output_file_path, gedi_small, gedi_profile)
            print('Resampling done')


def gedi_sentinel(path_gedi_resampled: Path, sentinel_feature: Path, parent_folder: Path, folder_name):
    print('New go')

    # Get absolute paths of files within the GEDI / validation directory
    gedi_paths = uf.get_abs_paths(path_gedi_resampled)

    # Get absolute paths of files within the Sentinel feature directory
    sentinel_paths = uf.get_abs_paths(sentinel_feature)

    # Filters GEDI dir and Sentinel dir such that they only contain symmetrical files based on dates
    gedi_files, sentinel_files = uf.only_common_dates(gedi_paths, sentinel_paths)

    # Specify output path
    output_path = parent_folder / f"{folder_name}_subset"

    print('gedi files', gedi_files)
    print('sentinel f', sentinel_files)

    # Loop through the GEDI files and sentinel feature files concomitantly to do the subset correctly
    for gedi, sentinel in zip(gedi_files, sentinel_files):
        print(f"Subsetting:{gedi} and {sentinel}")

        # Read GEDI data
        gedi_data, gedi_profile = uf.read_GeoTiff(gedi)

        # Read Sentinel feature
        sentinel_data, sentinel_profile = uf.read_GeoTiff(sentinel)

        # Create a mask for GEDI data
        mask = np.isnan(gedi_data)

        # Get the intersection of the given array and the mask
        intersection = np.copy(sentinel_data)
        intersection[mask] = np.nan

        # Create new file name and attach it to the output path
        to_rename = f"{sentinel.split('/')[6].rsplit('.')[0]}_resampled.tif"
        path_subsetted = output_path / to_rename

        # Write intersection
        with rio.open(path_subsetted, 'w', **sentinel_profile) as dst:
            dst.write(intersection)


# gedi_sentinel(GEDI_RESAMPLED, PATH_PROJECT / 'B1', PATH_PROJECT, 'B1')

# if __name__ == "__main__":
# eliminate corrupted files GEDI
# uf.filter_corrupted_null(PATH_GEDI)
# eliminate corrupted files bands / indexes
# for feat in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
#              'B12', 'ndvi', 'ndre', 'ndmi', 'evi']:
#     uf.filter_corrupted_null(PATH_PROJECT / feat)
# one-time run so that GEDI files get resampled once
# gedi_resolution(PATH_GEDI, INPUT_FILE_RESOLUTION_PATH, GEDI_RESAMPLED)
# subset sentinel bands
for dir in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
            'B12', 'ndvi', 'ndre', 'ndmi', 'evi']:
    sentinel_feat = PATH_PROJECT / dir
    gedi_sentinel(GEDI_RESAMPLED, sentinel_feat, PATH_PROJECT, dir)
