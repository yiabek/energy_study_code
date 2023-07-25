""" potential_calc.py

This takes windspeed and SSRD and prepares them for
energy potential calculation
"""

## Import packages
import numpy as np
from sys import exit

### Bring in NumPy arrays
windspeed = np.load("/data/singh/yianna/nparrays/ws_daily_array_annual_1980_2021.npy")
ssrd = np.load("/data/singh/yianna/nparrays/ssrd_array_Annual_1980_2021.npy")


### Change to time, lat, lon
windspeed = np.transpose(windspeed,(2,1,0)) 
ssrd = np.transpose(ssrd,(2,1,0))

### Ensure arrays are equal so code runs properly
numdays = int(windspeed.shape[0]/24)


### Create empty arrays 
maxwindspeed = np.empty([0, windspeed.shape[1], windspeed.shape[2]])  
ssrdsum = np.empty([0,ssrd.shape[1],ssrd.shape[2]])

### Loop to find max windspeed and cumulative ssrd
for day in np.arange(0, numdays):

    ### SSRD first....
    begind = day*24
    endind = begind+24
    dailyssrdsum = np.nansum(ssrd[begind:endind,:,:], axis=0)
    dailyssrdsum = dailyssrdsum.reshape(1,dailyssrdsum.shape[0],dailyssrdsum.shape[1])
    ssrdsum = np.concatenate([ssrdsum,dailyssrdsum])

    ### ...and then wind
    dailymax = np.nanmax(windspeed[begind:endind,:,:], axis=0)
    dailymax = dailymax.reshape(1, dailymax.shape[0], dailymax.shape[1])
    maxwindspeed = np.concatenate([maxwindspeed, dailymax])
    
### Transpose back to lon, lat, time
maxwindspeed = np.transpose(maxwindspeed,(2,1,0))
ssrdsum = np.transpose(ssrdsum,(2,1,0))
       
### Save numpy  arrays
np.save("/data/singh/yianna/nparrays/windspeed_max_daily_1980_2021.npy",maxwindspeed)
np.save("/data/singh/yianna/nparrays/ssrdsum_daily_1980_2021.npy",ssrdsum)
