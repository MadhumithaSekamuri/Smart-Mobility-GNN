'''
AUTHOR: Henry Saltzman

README

This script runs with the terminal command
python3 add_hex.py input_file output_file resolution

where input_file is the csv file containing accident data
        output_file is the name of the csv file which will contain accident data with h3 regions added
        resolution is the desired resolution of the h3 regions

'''

import h3
import pandas as pd
import sys

def geo_h3(row, res):
    return h3.geo_to_h3(lat = row['Start_Lat'], lng = row['Start_Lng'], resolution = res)

def geo_h3_2(row, res):
    return h3.geo_to_h3(lat = row['latitude'], lng = row['longitude'], resolution = res)

def get_neighbors(row):
    region = row['Region']
    neighbors = h3.k_ring(region, 1)
    neighbor_columns = {}
    for i, neighbor in enumerate(neighbors):
        neighbor_columns[f'Neighbor_{i+1}'] = neighbor
    return pd.Series(neighbor_columns)

def get_centroid(row):
    region = row['Region']
    centroid = h3.h3_to_geo(region)
    return pd.Series({'Centroid_Lat': centroid[0], 'Centroid_Lng': centroid[1]})

def join_demographic(accidents, demographic_name):
    demographics = pd.read_csv(demographic_name)
    demographics['Region'] = demographics.apply(lambda r : geo_h3_2(r, resolution), axis = 1)
    accidents = accidents.merge(demographics, on='Region', how='left')
    return accidents

input_file = sys.argv[1]
output_file = sys.argv[2]
resolution = int(sys.argv[3])

accidents = pd.read_csv(input_file)
accidents['Region'] = accidents.apply(lambda r : geo_h3(r, resolution), axis = 1)
accidents[['Centroid_Lat', 'Centroid_Lng']] = accidents.apply(lambda r : get_centroid(r), axis = 1)

neighbors = accidents.apply(lambda row : get_neighbors(row), axis=1)

accidents = pd.concat([accidents, neighbors], axis=1)


accidents = join_demographic(accidents, 'demographic_data.csv')
accidents.to_csv(output_file, index=False)


