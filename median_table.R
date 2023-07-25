###
## This script creates a CSV of medians
## for WHCE 
###

## Import packages
library(tidyverse)
library(purrr)
library(dplyr)

## Read in data
Predom.Event <- read_csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")

## Extract extreme events
Extreme.Events <- Predom.Event %>%
  filter(
    Event.Type != "Non-Extreme" &
        Season == 'DJF' & Event.Type == 'Cold' |
          Season == 'JJA' & Event.Type == 'Hot'
  )

## Calculate Duration (number of consecutive days of extreme temperature events)
duration <- Extreme.Events %>%
  mutate(Date = as.Date(Date)) %>%
  group_by(NERC.Region, Season, Event.Type, Year) %>%
  mutate(Event.ID = cumsum(c(1, diff(Date) > 1))) %>%
  group_by(NERC.Region, Season, Event.Type, Year, Event.ID) %>%
  summarize(Duration = as.numeric(difftime(max(Date), min(Date), 
                                           units = "days")) + 1,
            Start.Date = min(Date),
            .groups = "drop") %>%
  select(-Event.ID)

## Calculate frequency
frequency <- Extreme.Events %>%
  group_by(Year, Season, NERC.Region) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Season, NERC.Region, fill = list(n = 0))


# Extent.Median <- Extreme.Events %>%
#   group_by(NERC.Region, Season) %>%
#   summarise(across(where(is.numeric), median))

### -------------- Calculate medians of characteristics -------------- ###
### Data are grouped by NERC region and season
## Extent
extent_medians <- Extreme.Events %>%
  group_by(NERC.Region, Season) %>%
  summarise(Extent.Median = median(Extent))

## Intensity
intensity_medians <- Extreme.Events %>%
  group_by(NERC.Region, Season) %>%
  summarise(Intensity.Median = median(Intensity))

## Cumulative intensity
HSCI_medians <- Extreme.Events %>%
  group_by(NERC.Region, Season) %>%
  summarise(HSCI.Median = median(HSCI))
  
## Frequency
frequency_medians <- frequency %>%
    group_by(NERC.Region, Season) %>%
  summarise(Frequency.Median = median(n))
  
## Duration
duration_medians <- duration %>%
    group_by(NERC.Region, Season) %>%
  summarise(Duration.Median = median(Duration))
  
## Join all dataframes together for CSV
list_df = list(extent_medians, intensity_medians, HSCI_medians,
               frequency_medians, duration_medians)

## Merge list on NERC Region and season
medians <- list_df %>% reduce(merge, by=c("NERC.Region", "Season"))

## Save to csv
write.csv(medians, "/Users/yiannabekris/Documents/energy_data/csv/WHCE_medians.csv")




