#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:25:38 2023

@author: yiannabekris

This script takes power outage data from 2012-2021 compiled by the U.S. Department of Environment (DOE) 
Office of Cybersecurity, Energy Security, and Emergency Response and finds outages classified as weather-related.
It also standardizes the names of the NERC regions as they may vary, and coerces the data into a format that
can be consolidated with the widespread hot and cold extreme data. It does this by ensuring each day that was
impacted by an outage is represented in the data as well as splitting the data by NERC region if more than one
NERC region was impacted by an event. It outputs a single CSV for each year.

"""

import pandas as pd
import numpy as np

## For flattening NumPy arrays
def flatten(list_of_arrays):
    
   flat_list = np.hstack(list_of_arrays).tolist()
   
   return flat_list

### Designate years
years = [*range(2012, 2022, 1)]

## Loop through each year to extract data
for year in years:

    ### Load file
    outages = pd.read_excel(f'/Users/yiannabekris/Documents/energy_data/{year}_Annual_Summary.xls', header = 1)
    

    ## If the NERC Region field is NA, do not include it
    outages = outages[outages['NERC Region'].notna()]
    
    ## Find weather-related outages
    weather_outages = outages.loc[outages['Event Type'].str.contains("Weather", case=False)]
    
    ## Find instances of old NERC region names or multiple listed
    start_dates = np.asarray(pd.to_datetime(weather_outages['Date Event Began'], format='%m/%d/%Y', errors='ignore'))
    end_dates = np.asarray(weather_outages['Date of Restoration'])
    nerc_column = np.asarray(weather_outages['NERC Region'])
    customers_col = np.asarray(weather_outages['Number of Customers Affected'])
    demand_loss = np.asarray(weather_outages['Demand Loss (MW)'])
    
    ## Remove whitespace from end of Unknown end dates
    end_dates[end_dates=='Unknown '] = 'Unknown'
    
    ## Initiate empty lists
    event_number = []
    outage_dates = []
    durations = []
    nerc_regions = []
    customers_affected = []
    demand_losses = []
    
    
    ## Loop through and fill in dates between start and end
    for ind in np.arange(0, len(start_dates)):
        
        ## If the end date is not unknown
        if end_dates[ind] != 'Unknown' and (start_dates[ind] != end_dates[ind]):
            end_date = pd.to_datetime(end_dates[ind], errors='ignore')
            datatime = pd.date_range(start=start_dates[ind], end=end_date)
            datatime = pd.to_datetime(datatime)
            dates=datatime.strftime("%Y/%m/%d")
            inds = np.repeat(ind, len(dates))
            customers = np.repeat(customers_col[ind], len(dates))
            nerc = np.repeat(nerc_column[ind], len(dates))
            demand = np.repeat(demand_loss[ind], len(dates))
            duration = np.repeat(len(dates), len(dates))
            
            ## Append to lists
            event_number.append(inds)
            outage_dates.append(dates)
            durations.append(duration)
            nerc_regions.append(nerc)
            customers_affected.append(customers)
            demand_losses.append(demand)
    
        else:
            datatime = start_dates[ind]
            datatime = pd.to_datetime(datatime)
            dates=datatime.strftime("%Y/%m/%d")
            durations.append(1)
            
            # Append to lists
            event_number.append(ind)
            outage_dates.append(dates)
            nerc_regions.append(nerc_column[ind])
            customers_affected.append(customers_col[ind])
            demand_losses.append(demand_loss[ind])
        
            
    
    ## Flatten each array for pandas dataframe
    event_flat = flatten(event_number)
    outage_flat = flatten(outage_dates)
    duration_flat = flatten(durations)
    nerc_flat = flatten(nerc_regions)
    customers_flat = flatten(customers_affected)
    demand_flat = flatten(demand_losses)
    
    
    ## Create pandas dataframe from data
    outage_df = pd.DataFrame({'Event.ID' : event_flat, 'Date': outage_flat, 'Duration': duration_flat,
                              'NERC.Region': nerc_flat, 'Customers.Impacted': customers_flat,
                              'Demand.Loss': demand_flat})
    
    ## Find instances of old NERC region names or multiple listed
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('FRCC', 'SERC')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('RF', 'RFC')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('SPP RE', 'MRO')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('SPP', 'MRO')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('RFCC', 'RFC')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('NP', 'NPCC')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('NPCCCC', 'NPCC')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace('/', ',')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace(' ', '')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.replace(';', ',')
    outage_df['NERC.Region'] = outage_df['NERC.Region'].str.split(',')
    
    ## Explode data 
    outages_exploded = outage_df.explode('NERC.Region')
    
    # print NERC Regions to make sure all were included
    print(outages_exploded['NERC.Region'].value_counts())
    
    ## Save to CSV for plotting in R
    outages_exploded.to_csv(f'/Users/yiannabekris/Documents/energy_data/csv/weather_outages/outages_{year}.csv')
    print('DONE!!!!!')
    
