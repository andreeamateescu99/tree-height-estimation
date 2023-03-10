import pandas as pd
from pathlib import Path
import os
import plotly.graph_objs as go

validation_ds_path = Path(__file__).parent / 'validation_datasets'
list_values = list()

fig = go.Figure()
for subfolder in os.listdir(validation_ds_path):
    if os.path.isdir(os.path.join(validation_ds_path, subfolder)):
        csv_path = os.path.join(validation_ds_path, subfolder, 'new_' + subfolder + '.csv')
        country = pd.read_csv(csv_path)
        land_cover_dict = country['LC1'].value_counts().to_dict()
        fig.add_trace(go.Histogram(x=list(land_cover_dict.keys()), y=list(land_cover_dict.values()), name=f"{subfolder}", histfunc='sum'))


fig.update_layout(title='Distributions of woodland covers 2018', xaxis_title='Different land covers', yaxis_title='Counts')

with open(f"validation_2018.html", "a") as f:
    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
