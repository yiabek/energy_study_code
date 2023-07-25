#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates wind and solar potential 

@author: yiannabekris
"""

## Import numpy
import numpy as np

### Can take a scalar or an array
## Returns wind power per turbine (kW) using the simplified
## formula as well as the formula specific to a type of wind turbine
def wind_potential(windspeed_hourly):
    
    ### Convert windspeed to floats in case it is not already
    windspeed_hourly = windspeed_hourly.astype(float)
    wind_power_density = (0.5 * 1.225 * 452.4 * windspeed_hourly**3)/1000 # convert from W to kW
    
    ### This is the power curve from the paper "A k-nearest neighbor..."
    W = windspeed_hourly ## rename for ease
    
    P = 634.228 - 1248.5 * W\
        + 999.57 * W**2 - 426.224 * W**3\
        + 105.617 * W**4 - 15.4587 * W**5\
        + 1.3223 * W**6 - 0.0609186 * W**7\
        + 0.00116265 * W**8
        
    ### Set to 0 or 2000 if wind speed is over or under the defined thresholds
    P[windspeed_hourly>=25.0] = 0
    P[windspeed_hourly<3] = 0
    P[(windspeed_hourly>=13)&(windspeed_hourly<25)] = 2000
    
    
    wind_power_density[windspeed_hourly>=25.0] = 0
    wind_power_density[windspeed_hourly<3] = 0
    wind_power_density[(windspeed_hourly>=13) & (windspeed_hourly<25)] = 2000

    return wind_power_density, P


### This calculates solar potential using downward surface solar radiation
### Inputs can either be scalars or arrays of air temperature and solar radiation
## Output is solar capacity factor and relative efficiency     
def solar_potential(airtemp, ssrd):
    print('Calculating solar potential.... ')
    
    ### Variables for calculating solar potential
    a = 4.2 * 10**-3
    b = -4.6 * 10**-3
    c1 = 0.033
    c2 = -0.0092
    T_noct = 48
    T_0 = 20
    G_0 = 800
    G_stc = 1000 # W m**-2
    T_stc = 25 ## module temperature in test conditions (C)
    G = ssrd / 86400 # 3600 ## in-plane irradiance 
    G_prime = G/G_stc
    T_mod = airtemp + (T_noct - T_0) * G/G_0
    delta_T_mod = T_mod - T_stc
        
    ### Calculate relative efficiency
    rel_efficiency = (1+a*delta_T_mod)\
                     * (1 + c1*np.log(G_prime) + c2 * (np.log(G_prime)**2)\
                     + b * delta_T_mod)
                         
                     
    ### Calculate capacity factor                    
    capacity_factor = rel_efficiency * G/G_stc
    
    return capacity_factor, rel_efficiency
