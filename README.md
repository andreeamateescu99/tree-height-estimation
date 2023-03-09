# tree-height-estimation
In this repository I will do a tree height prediction based on Sentinel-1 and Sentinel-2 data.

Architecture
1. Reading tif files from Google Cloud
2. Reading validation datasets
3. Exporting polygon for less than 100 ha per export, in order to be fed to the query of Meteory