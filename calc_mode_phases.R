#### mode_phases.R
## This script takes data of the following modes of climate variability:
## Arctic Oscillation (AO), North Atlantic Oscillation (NAO), 
## El Niño-Southern Oscillation (ENSO), and Pacific/North American (PNA) pattern
## and defines the positive and negative phases for each mode as months
## with standardized anomalies over 0.5 or under -0.5. 
## This then saves the output as a CSV, one for each mode.
####

## Import packages
library(tidyverse)
library(snakecase)
library(dplyr)

### ============= Preparing data for KS test ============= ### 
## Read in indices data
ao <- read.csv('/Users/yiannabekris/Documents/energy_data/clim_indices/ao.csv')
nao <- read.csv('/Users/yiannabekris/Documents/energy_data/clim_indices/nao.csv')
nino34 <- read.csv('/Users/yiannabekris/Documents/energy_data/clim_indices/nino34.csv')
pna <- read.csv('/Users/yiannabekris/Documents/energy_data/clim_indices/pna.csv')

## Format columns so they are uniform
for(i in colnames(nao)){
  colnames(nao)[which(colnames(nao)==i)] = to_upper_camel_case(i)
}

for(i in colnames(nino34)){
  colnames(nino34)[which(colnames(nino34)==i)] = to_upper_camel_case(i)
}

for(i in colnames(pna)){
  colnames(pna)[which(colnames(pna)==i)] = to_upper_camel_case(i)
}

## Rename columns
nino34 <- nino34[c("Yr", "Mon", "Anom3")]
nino34 <- nino34 %>% rename("Year"="Yr","Month"="Mon","Index"="Anom3")


## Index to be analyzed
index <- nino34
index_name <- 'nino34'

## Add seasons column to each mode of climate variability dataframe
ao <- ao %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

nao <- nao %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

nino34 <- nino34 %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

pna <- pna %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

## Mean of index for the season ## Comment out when using monthly values
# ao <- ao %>% aggregate(Index ~ Year + Season, mean)
# nao <- nao %>% aggregate(Index ~ Year + Season, mean)
# nino34 <- nino34 %>% aggregate(Index ~ Year + Season, mean)
# pna <- pna %>% aggregate(Index ~ Year + Season, mean)

## Add positive and negative column for each index
ao <- ao %>% mutate(Mode = 'AO')
nao <- nao %>% mutate(Mode = 'NAO')
nino34 <- nino34 %>% mutate(Mode = 'NINO3.4')
pna <- pna %>% mutate(Mode = 'PNA')

ao <- ao %>% mutate(Phase = case_when(
  Index > 0.5 ~ 'Positive',
  Index < -0.5 ~ 'Negative',
  Index <= 0.5 & Index >= -0.5 ~ 'Neutral'
))

nao <- nao %>% mutate(Phase = case_when(
  Index > 0.5 ~ 'Positive',
  Index < -0.5 ~ 'Negative',
  Index <= 0.5 & Index >= -0.5 ~ 'Neutral'
))

nino34 <- nino34 %>% mutate(Phase = case_when(
  Index > 0.5 ~ 'Positive',
  Index < -0.5 ~ 'Negative',
  Index <= 0.5 & Index >= -0.5 ~ 'Neutral'
))

pna <- pna %>% mutate(Phase = case_when(
  Index > 0.5 ~ 'Positive',
  Index < -0.5 ~ 'Negative',
  Index <= 0.5 & Index >= -0.5 ~ 'Neutral'
))

## Save each CSV
write.csv(ao, "/Users/yiannabekris/Documents/energy_data/clim_indices/ao_0_5_thresh.csv", row.names=FALSE)
write.csv(nao, "/Users/yiannabekris/Documents/energy_data/clim_indices/nao_0_5_thresh.csv", row.names=FALSE)
write.csv(nino34, "/Users/yiannabekris/Documents/energy_data/clim_indices/nino34_0_5_thresh.csv", row.names=FALSE)
write.csv(pna, "/Users/yiannabekris/Documents/energy_data/clim_indices/pna_0_5_thresh.csv", row.names=FALSE)

