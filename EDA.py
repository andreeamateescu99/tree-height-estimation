import pandas as pd
from pathlib import Path
import os
import plotly.graph_objs as go
import plotly.express as px

validation_ds_path = Path(__file__).parent / 'validation_datasets'

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


def plot_comparative_validation_attributes(validation_folder_path):
    fig = go.Figure()
    for subfolder in os.listdir(validation_ds_path):
        if os.path.isdir(os.path.join(validation_ds_path, subfolder)):
            csv_path = os.path.join(validation_folder_path, subfolder, 'new_' + subfolder + '.csv')
            country = pd.read_csv(csv_path)
            land_cover_dict = country['LC1'].value_counts().to_dict()
            fig.add_trace(
                go.Histogram(x=list(land_cover_dict.keys()), y=list(land_cover_dict.values()), name=f"{subfolder}",
                             histfunc='sum'))
    fig.update_layout(title='Distributions of woodland covers 2018', xaxis_title='Different land covers',
                      yaxis_title='Counts')
    with open(f"validation_2018.html", "a") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

# plot geographical area
path_country = pd.read_csv('/Users/andreeamateescu/PycharmProjects/tree-height-estimation/validation_datasets/France/new_France.csv')
mapbox_token = 'pk.eyJ1IjoiYW5kcmVtYXQiLCJhIjoiY2xmOGR6cGloMGJoNzNxcW1kNjF5eXR3eCJ9.8PsmFVMU1AwgtsFupnWD7w'
fig = px.scatter_mapbox(path_country, lat='TH_LAT', lon='TH_LONG', zoom=10,
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                        height=600, mapbox_style='mapbox://styles/mapbox/streets-v11',
                        opacity=0.8)

fig.update_layout(mapbox={'accesstoken': mapbox_token})

with open(f"geo_validation_2018.html", "a") as f:
    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))