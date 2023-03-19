import pandas as pd
from pathlib import Path
import os

validation_ds_path = Path(__file__).parent / 'validation_datasets'


def read_all_validation_files(validation_folder_path):
    """This function reads all validation datasets, and saves the validation datasets concerning
    latitutde, longitude, land cover, land use information for woodland ares"""
    for subfolder in os.listdir(validation_folder_path):
        if os.path.isdir(os.path.join(validation_ds_path, subfolder)):
            csv_path = os.path.join(validation_ds_path, subfolder, subfolder + '.csv')
            country = pd.read_csv(csv_path, sep=',')
            subset_land_cover = country[
                ['POINT_ID', 'TH_LAT', 'TH_LONG', 'LC1', 'LC2', 'LU1', 'TREE_HEIGHT_SURVEY', 'TREE_HEIGHT_MATURITY']]
            subset_land_cover = subset_land_cover.loc[
                subset_land_cover['LC1'].isin(['C10', 'C21', 'C22', 'C23', 'C31', 'C32', 'C33'])]

            new_csv_path = os.path.join(validation_ds_path, subfolder, 'new_' + subfolder + '.csv')
            subset_land_cover.to_csv(new_csv_path, index=False)


