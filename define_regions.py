import pandas as pd
import h3
import sys

'''
AUTHOR: Henry Saltzman

README

This script runs with the terminal command
python3 define_regions.py county state output_file

where county is the desired county to find regions
        state is the desired state to find regions
        output_file is the name of the csv file which will contain all regions for this county
'''

def define_regions(county, state):
    counties = pd.read_csv('US_County_Boundingboxes.csv')

    this_county = counties[(counties.COUNTY_NAME == county) & (counties.STATE_NAME == state)]
    min_lat = this_county['xmin'].values[0]
    max_lat = this_county['xmax'].values[0]
    min_long = this_county['ymin'].values[0]
    max_long = this_county['ymax'].values[0]

    lat =  min_lat
    regions = set()

    while lat < max_lat:
        long = min_long
        while long < max_long:
            region = h3.geo_to_h3(lat = lat, lng = long, resolution = 7)
            if region not in regions:
                regions.add(region)
            long += 0.00001
        lat += 0.00001
    
    return regions

county = sys.argv[1]
state = sys.argv[2]
output_file = sys.argv[3]

regions = define_regions(county, state)

regions_df = pd.DataFrame(list(regions), columns=['regions'])

regions_df.to_csv(output_file)