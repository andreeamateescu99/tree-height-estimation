import pandas as pd
from pathlib import Path

COUNTRY = 'France'

VALIDATION_FILE = 'FR_2018_20200213.csv'

csv_file = Path(__file__).parent / 'validation_datasets' / COUNTRY / VALIDATION_FILE
fr = pd.read_csv(csv_file, sep=',')
columns = fr.columns
subset_land_cover = fr[['LC1', 'LC2', 'LU1']]
for column in ['LC1', 'LC2', 'LU1']:
    print(column + ' name is:')
    coniferous = subset_land_cover[column][subset_land_cover[column].isin(['C10', 'C21', 'C23', 'C31', 'C32', 'C33'])]
    counts = coniferous.value_counts()

# For all the points that have coniferous, retrieve them and make a new db

# make statistics of those points

# make some checks, using plotly and the latitude and longitude, of how extended this dataset is

# download and compare 2018 data for other 2 countries that are nearby France and / or seem to share the same type of forest

# check in LUCAS db what other interesting data points you can get

# check well to what other countries the study can be generalised

# make some notes related to potential downsides of the evaluation data

# check with the dates as well




# print(value_counts_dict)