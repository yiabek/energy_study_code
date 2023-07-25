##### Permutation Test
## This script conducts a permutation test for solar and wind potential
## and outputs the results to CSVs to be used for plotting
#
# 1) Calculate difference in means
# 2) Combine the 2 distributions
# 3) Sample with replacement from the new distribution (length of dist1 and dist2) (do n times)
# 4) Calculate difference between difference between new dist1 and dist2 (do n times)
# 5) Calculate the 5th and 95th percentile of the differences 
# 6) See if the actual difference is outside of the bounds of the 5th and 95th percentile
#  -- if it is outside of those bounds it means it is significant 
# output of the function should be significant or not significant
##
##
####

### Import packages
library(dplyr)
library(tidyr)
library(permute)

### Read in data
predom_event <- read.csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")

### Number of repeats
repeats <- 1000

## Classify "Cold" and "Hot" as "Extreme"
predom_event$Event.Type[predom_event$Event.Type=="Cold" | predom_event$Event.Type=="Hot"] <- "Extreme"

## Difference in medians between non-extreme and extreme days
## For solar and wind potential
## Grouped by NERC Region, season, and event type
median_diff <- predom_event %>%
  group_by(NERC.Region, Season, Event.Type) %>%
  summarise(WP.Median = median(Wind.Potential),
            SP.Median = median(Solar.Potential)) %>%
  mutate(WP.Median.Diff = diff(WP.Median, 1),
         SP.Median.Diff = diff(SP.Median, 1))

## Subset to only the necessary columns
wp_diff <- median_diff[,c(1,2,6)]

## Remove duplicates of the differences as they will be the same
## for extreme and non-extreme days
wp_diff <- wp_diff[!duplicated(wp_diff),]

## Join the event dataframe and the difference in medians
## for the permutation tests
event_permute <- left_join(predom_event, median_diff,
                           by = c("NERC.Region", "Season", "Event.Type"))

## Initiate empty tibble to store all differences in medians for wind potential
wp_median_diffs <- tibble()

## Conduct the permutation test and repeat as many times as
## delineated by the variables repeats
for(i in 1:repeats){

  ## Sum the number of extreme and non-extreme observations
  ## then take the median for wind potential,
  ## grouped by NERC region and season
  wp_medians <- predom_event %>%
    group_by(NERC.Region, Season) %>%
    summarise(n_ex = sum(Event.Type=="Extreme"),
              n_nonex = sum(Event.Type=="Non-Extreme"),
              ex_median = median(sample(Wind.Potential, n_ex, TRUE)),
              nonex_median = median(sample(Wind.Potential, n_nonex, TRUE)))

  ## Find the difference of medians
  wp_median_diff <- wp_medians %>%
    group_by(NERC.Region, Season) %>%
    summarise(median_diff = sum(ex_median - nonex_median))
  
  ## Bind to tibble of the repeats of the differences in medians
  wp_median_diffs <- rbind(wp_median_diffs, wp_median_diff)
  
}

## Find the 5th and 95th percentile of the difference in medians for wind potential
wp_quants <- wp_median_diffs %>%
  group_by(NERC.Region, Season) %>%
  summarise(quant_5 = quantile(median_diff, probs = 0.05),
            quant_95 = quantile(median_diff, probs = 0.95))

## Join the original difference in medians to the 5th and 95th percentile dataframe
wp_quants_diff <- full_join(wp_quants, wp_diff,
                            by = c("NERC.Region", "Season"))

## Create a column which shows significance. 
## If the median difference in medians
## is less than the 5th percentile or greater than the 95th percentile
## the distributions are considered significantly different from each other
wp_perm_sig <- wp_quants_diff %>% 
  mutate(
    sig = ifelse(WP.Median.Diff < quant_5 | WP.Median.Diff > quant_95, "*", "")
  )


### Permutation test for solar potential
## Subset to only the necessary columns
sp_diff <- median_diff[,c(1,2,7)]

## Remove duplicates of the differences as they will be the same
## for extreme and non-extreme days
sp_diff <- sp_diff[!duplicated(sp_diff),]

## Initiate empty tibble to store all differences in medians
sp_median_diffs <- tibble()

## Conduct the permutation test and repeat as many times as
## delineated by the variables repeats
for(i in 1:repeats){

  ## Sum the number of extreme and non-extreme observations
  ## then take the median for solar potential,
  ## grouped by NERC region and season
  median_extreme <- predom_event %>%
    group_by(NERC.Region, Season) %>%
    summarise(n_values = sum(Event.Type=="Extreme"),
              ex_median = median(sample(Solar.Potential, n_values, TRUE)))
  
  median_nonextreme <- predom_event %>%
    group_by(NERC.Region, Season) %>%
    summarise(n_values = sum(Event.Type=="Non-Extreme"),
              nonex_median = median(sample(Solar.Potential, n_values, TRUE)))
  
  ## Join median dataframes together
  medians_join <- full_join(median_extreme, median_nonextreme,
                            by = c("NERC.Region", "Season"))
  
  ## Find the difference in medians for solar potential
  median_diff_sp <- medians_join %>%
    group_by(NERC.Region, Season) %>%
    summarise(median_diff = sum(ex_median - nonex_median))
  
  ## Bind each repeat to the tiblle of differences in medians
  sp_median_diffs <- rbind(sp_median_diffs, median_diff_sp)
  

}

## Find the 5th and 95th percentile of the difference in medians for solar potential
sp_quants_diff <- sp_median_diffs %>%
  group_by(NERC.Region, Season) %>%
  summarise(quant_5 = quantile(median_diff, probs = 0.05),
            quant_95 = quantile(median_diff, probs = 0.95))

## Join the quantiles and differences in medians together
sp_quants_diff <- full_join(sp_quants_diff, sp_diff,
                            by = c("NERC.Region", "Season"))

## Create a column which shows significance. 
## If the median difference in medians
## is less than the 5th percentile or greater than the 95th percentile
## the distributions are considered significantly different from each other
sp_perm_sig <- sp_quants_diff %>% 
  mutate(
    sig = ifelse(SP.Median.Diff < quant_5 | SP.Median.Diff > quant_95, "*", "")
  )

### Save the output to a CSV for plotting
write_csv(wp_perm_sig, "/Users/yiannabekris/Documents/energy_data/csv/wind_potential_perm_test_1000.csv")
write_csv(sp_perm_sig, "/Users/yiannabekris/Documents/energy_data/csv/solar_potential_perm_test_1000.csv")
