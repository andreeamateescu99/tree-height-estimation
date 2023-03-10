import pandas as pd
from pathlib import Path
import os

validation_ds_path = Path(__file__).parent / 'validation_datasets'

for subfolder in os.listdir(validation_ds_path):
    if os.path.isdir(os.path.join(validation_ds_path, subfolder)):
        csv_path = os.path.join(validation_ds_path, subfolder, subfolder + '.csv')
        country = pd.read_csv(csv_path, sep=',')
        subset_land_cover = country[
                ['POINT_ID', 'TH_LAT', 'TH_LONG', 'LC1', 'LC2', 'LU1', 'TREE_HEIGHT_SURVEY', 'TREE_HEIGHT_MATURITY']]
        subset_land_cover = subset_land_cover.loc[
            subset_land_cover['LC1'].isin(['C10', 'C21', 'C22', 'C23', 'C31', 'C32', 'C33'])]

        new_csv_path = os.path.join(validation_ds_path, subfolder, 'new_' + subfolder + '.csv')
        subset_land_cover.to_csv(new_csv_path, index=False)

# 2 make statistics of those points including

# 3 make some checks, using plotly and the latitude and longitude, of how extended this dataset is

# * download and compare 2018 data for other 2 countries that are nearby France and / or seem to share the same type of forest

# * check in LUCAS db what other interesting data points you can get

# 4 check well to what other countries the study can be generalised

# 6 make some notes related to potential downsides of the evaluation data

# 4 check with the dates as well


# print(value_counts_dict)
