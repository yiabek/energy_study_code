### find_whce_run_lm.R
## This takes the data wrangled by energy_csv.py
## and outputs the following:
## a CSV with each day defined as extreme or non-extreme
## with all characteristics calculated and energy potential computed
## CSVs with linear model results for 
## frequency, mean extent, mean cumulative intensity, and mean duration
## A table with p-values and percent change for annotating figure 2
###

## Import packages
library(broom)
library(dplyr)
library(lubridate)

## Load data of annual events
annual <- read.csv('/Users/yiannabekris/Documents/energy_data/csv/annual_metrics_1_5_anom.csv')

## Filenames for saving CSVs
predom_event_filename <- '/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv'
frequency_filename <- '/Users/yiannabekris/Documents/energy_data/csv/frequency_trends_1_5_80.csv'
mean_extent_filename <- '/Users/yiannabekris/Documents/energy_data/csv/mean_extent_trends_1_5_80.csv'
mean_CI_filename <- '/Users/yiannabekris/Documents/energy_data/csv/mean_CI_trends_1_5_80.csv'
mean_duration_filename <- '/Users/yiannabekris/Documents/energy_data/csv/mean_duration_trends_1_5_80.csv'

## Set extent threshold; must be between 0 and 1
threshold <- 0.8
anom_thresh <- 1.5

### Create month, year, and day columns
annual <- annual %>%
  mutate(Year = lubridate::year(Date),
         Month = lubridate::month(Date),
         Day = lubridate::day(Date)) 

### Add seasons column
annual <- annual %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(12, 1, 2)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

## Filter by season
annual <- annual %>% filter(Season %in% c("DJF", "JJA"))

## Convert December to the following year, so it is clustered with 
## the following January and February
annual <- annual %>% mutate(
    Year = ifelse(
      Month == 12, Year + 1, Year
    )
  )

## Remove January and February 1980 because there is no December 1979
annual <- annual %>% filter(
     !Month %in% c(1, 2) | !Year %in% 1980
 )

## Remove December 2022
annual <- annual %>% filter(
  Year != 2022
)

## Change "Hot" events in DJF to "Non-Extreme"
## and "Cold" events in JJA to "Non-Extreme"
annual$Event.Type[annual$Event.Type == "Hot" & annual$Season == "DJF"] <- "Non-Extreme"
annual$Event.Type[annual$Event.Type == "Cold" & annual$Season == "JJA"] <- "Non-Extreme"


## Calculate 80th percentile of extent (threshold can be changed)
thresh <- annual %>%
  group_by(NERC.Region, Season, Event.Type) %>%
  summarize(quant = quantile(Extent, probs=threshold, na.rm=TRUE))

## Filter by event type 
cthresh <- thresh %>% filter(Event.Type == 'Cold')
hthresh <- thresh %>% filter(Event.Type == 'Hot')

## Change anything below the percentile threshold for cold extent to a non-extreme event
for(i in 0:length(cthresh$quant)){
  cthreshold <- cthresh$quant[i]
  annual$Event.Type[annual$Event.Type == 'Cold' & annual$NERC.Region == cthresh$NERC.Region[i] & 
                      cthresh$Season == cthresh$Season[i]& annual$Extent < cthreshold] <- 'Non-Extreme'
}

## Change anything below the percentile threshold for hot extent to a non-extreme event
for(i in 0:length(hthresh$quant)){
  hthreshold <- hthresh$quant[i]
  annual$Event.Type[annual$Event.Type == 'Hot' & annual$NERC.Region == hthresh$NERC.Region[i] & 
                      hthresh$Season == hthresh$Season[i]& annual$Extent < hthreshold] <- 'Non-Extreme'
}

## Change non-extreme event extent to 0 so extreme events
## are not misclassified as non-extreme
annual$Extent[annual$Event.Type == 'Non-Extreme'] <- 0

## Find the predominant event
Predom.Event <- annual %>% 
  na.omit() %>%
  group_by(NERC.Region, Date) %>% 
  slice(which.max(Extent)) 

## If extent is 0, change to NA
Predom.Event$Extent[Predom.Event$Extent == 0.00000] <- NA

## Write the predominant events (hot, cold, or non-extreme based on the thresholds)
## to a CSV
write.csv(Predom.Event, predom_event_filename)

## Generate a vector of the climatology years (1991-2020)
clim_years <- 1991:2020

## Subset data to extreme events
Extreme.Events <- Predom.Event %>%
  filter(
    Event.Type != "Non-Extreme"
  )

## Subset to the climatology years for calculating climatologies
Extreme.Event.Clim <- Extreme.Events %>% 
  filter(
    Year %in% clim_years
    )


## Calculate duration (number of consecutive days of extreme temperature events)
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


### Create frequency models 
## Calculate annual averages
Frequency.Avg <- Extreme.Events %>%
  group_by(Year, Season, NERC.Region) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Season, NERC.Region, fill = list(n = 0))

## Rename the column ouput by above code to "Frequency"
Frequency.Avg <- rename(Frequency.Avg, "Frequency" = "n")

## Remove DJF 1980 which was added by the frequency calculation
Frequency.Avg <- Frequency.Avg %>%
  filter(!Season %in% c("DJF") | !Year %in% 1980)

## Run linear model for frequency as explained by year
freq_mods <- Frequency.Avg %>%
  group_by(NERC.Region, Season) %>%
  group_modify(
    ~ tidy(lm(Frequency ~ Year, data = .)),
  ) %>% 
  ungroup()

## Rename "estimate" column as "Trend"
freq_mods <- rename(freq_mods, "Trend" = "estimate")

## Extract the coefficient calculated by the linear model function
freq_coeff <- freq_mods %>% filter(term=="Year")

## Calculate the frequency average over 1991-2020 climatology period
freq_clim <- Frequency.Avg %>% filter(Year %in% clim_years)
Frequency.Clim <- aggregate(Frequency ~ Season + NERC.Region, freq_clim, mean, na.rm=TRUE)

## Rename column it is not confused with the other "Frequency" column
Frequency.Clim <- Frequency.Clim %>% rename("Frequency.Clim" = "Frequency")

## Join
frequency_all <- freq_coeff %>% left_join(Frequency.Clim, by = c("NERC.Region","Season")) 

## Calculate percent change
freq_coeff$`Percent Change` <- ((frequency_all$Trend) * 42 / frequency_all$Frequency.Clim) * 100

## Save to csv
write.csv(freq_coeff, frequency_filename)

### Extent models
## Mean extent
## Calculate annual averages
Extent.Avg <- aggregate(Extent ~ Season + NERC.Region + Event.Type + Year, Extreme.Events, mean, na.rm=TRUE)

## Calculate extent average over 1991-2020 climatology period
Extent.Clim <- aggregate(Extent ~ Season + NERC.Region + Event.Type, Extreme.Event.Clim, mean, na.rm=TRUE)

## Rename columns
Extent.Avg <- rename(Extent.Avg, "Extent.Mean" = "Extent")
Extent.Clim <- Extent.Clim %>% rename("Extent.Clim" = "Extent")

## Run linear model for mean extent as explained by year
mean_extent_mods <- Extent.Avg %>%
  group_by(NERC.Region, Event.Type, Season) %>%
  group_modify(
    ~ tidy(lm(Extent.Mean ~ Year, data = .))
  ) %>% ungroup()

## Rename "estimate" column as "Trend"
mean_extent_mods <- rename(mean_extent_mods, "Trend" = "estimate")

## Extract the coefficient calculated by the linear model function
mean_extent_coeff <- mean_extent_mods %>% filter(term=="Year")

## Join
mean_extent_all <- mean_extent_coeff %>% left_join(Extent.Clim, by = c("NERC.Region","Season")) 

## Calculate percent change
mean_extent_coeff$`Percent Change` <- ((mean_extent_all$Trend) * 42 / mean_extent_all$Extent.Clim) * 100

## Save to csv
write.csv(mean_extent_coeff, mean_extent_filename)


### Cuulative Intensity (CI)
## Mean CI
## Calculate annual averages 
CI.Avg <- aggregate(CI ~ Season + NERC.Region + Event.Type + Year, 
                      Extreme.Events, mean, na.rm=TRUE)

## Calculate CI average over 1991-2020 climatology period
CI.Clim <- aggregate(CI ~ Season + NERC.Region + Event.Type, Extreme.Event.Clim, mean, na.rm=TRUE)

## Rename columns
CI.Avg <- rename(CI.Avg, "CI.Mean" = "CI")
CI.Clim <- CI.Clim %>% rename("CI.Clim" = "CI")

## Run linear model for mean CI as explained by year
mean_CI_mods <- CI.Avg %>%
  group_by(NERC.Region, Season) %>%
  group_modify(
    ~ tidy(lm(CI.Mean ~ Year, data = .))
  ) %>% ungroup()

## Rename "estimate" column as "Trend"
mean_CI_mods <- rename(mean_CI_mods, "Trend" = "estimate")

## Extract the coefficient calculated by the linear model function
mean_CI_coeff <- mean_CI_mods %>% filter(term=="Year")

## Join
mean_CI_all <- mean_CI_coeff %>% left_join(CI.Clim, by = c("NERC.Region","Season")) 

## Calculate percent change
mean_CI_coeff$`Percent Change` <- ((mean_CI_all$Trend) * 42 / mean_CI_all$CI.Clim) * 100

## Save to csv
write.csv(mean_CI_coeff, mean_CI_filename)

### Duration
## Calculate annual averages
Duration.Avg <- aggregate(Duration ~ Season + NERC.Region + Event.Type + Year, duration, mean)

## Calculate duration average over 1991-2020 climatology period
Duration.Clim.Years <- duration %>% filter(Year %in% clim_years)
Duration.Clim <- aggregate(Duration ~ Season + NERC.Region + Event.Type, Duration.Clim.Years, mean)

## Run linear model for mean duration as explained by year
mean_duration_mods <- Duration.Avg %>%
  group_by(NERC.Region, Event.Type, Season) %>%
  group_modify(
    ~ tidy(lm(Duration ~ Year, data = .))
  ) %>% ungroup()

## Rename "estimate" column as "Trend"
mean_duration_mods <- rename(mean_duration_mods, "Trend" = "estimate")

## Extract the coefficient calculated by the linear model function
mean_duration_coeff <- mean_duration_mods %>% filter(term=="Year")

## Rename columns
Duration.Avg <- rename(Duration.Avg, "Duration.Mean" = "Duration")
Duration.Clim <- Duration.Clim %>% rename("Duration.Clim" = "Duration")

## Join
mean_duration_all <- mean_duration_coeff %>% left_join(Duration.Clim, by = c("NERC.Region","Season")) 

## Calculate percent change
mean_duration_coeff$`Percent Change` <- ((mean_duration_all$Trend) * 42 / mean_duration_all$Duration.Clim) * 100

## Save to csv
write.csv(mean_duration_coeff, mean_duration_filename)


## Subset columns for the significance table for plotting figure 2
freq_sub <- freq_coeff %>% select(NERC.Region, Season, p.value)
mean_extent_sub <- mean_extent_coeff %>% select(NERC.Region, Season, p.value)
mean_duration_sub <- mean_duration_coeff %>% select(NERC.Region, Season, p.value)
mean_CI_sub <- mean_CI_coeff %>% select(NERC.Region, Season, p.value)

## Add in a metric column for selection during plotting figure 2
freq_sub$Metric <- 'Frequency'
mean_extent_sub$Metric <- 'Extent'
mean_duration_sub$Metric <- 'Duration'
mean_CI_sub$Metric <- 'CI'

## Bind each dataframe together to create the significance table
mean_sig_table <- rbind(freq_sub, mean_extent_sub, mean_duration_sub, mean_CI_sub)

## Save the table as a CSV
write.csv(mean_sig_table, '/Users/yiannabekris/Documents/energy_data/csv/mean_sig_table_1_5_80.csv')

