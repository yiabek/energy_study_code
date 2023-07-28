#### figure3.R
## This script creates energy potential plots for Figure 3.
## These plots include probability distributions for wind capacity factor
## and solar capacity factor as well as their medians, 
## grouped by NERC region, season, and event type.
## The significance results of the permutation tests are plotted
## with an asterisk.
## The permutation test results are saved from the permutation_test.R script.
###

## Import packages
library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggridges)
library(cowplot)

## Read csv with event data
predom_events <- read_csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")
sp_perm_test <- read.csv("/Users/yiannabekris/Documents/energy_data/csv/solar_potential_perm_test_1000.csv")
wp_perm_test <- read.csv("/Users/yiannabekris/Documents/energy_data/csv/wind_potential_perm_test_1000.csv")

## Create wind dataframe
wind <- predom_events

## Convert hot and non-extreme days to "other" in DJF
wind$Event.Type[wind$Event.Type=='Hot' &
                  wind$Season=='DJF' | wind$Event.Type=='Non-Extreme'] <- 'Other'

## Convert cold and non-extreme days to "other" in JJA
wind$Event.Type[wind$Event.Type=='Cold' &
                  wind$Season=='JJA' | wind$Event.Type=='Non-Extreme'] <- 'Other'

## Set ordering of NERC regions on plot
## and make sure they are factors in the data so they plot right
levels <- c("RFC", "SERC", "TRE", "NPCC", "MRO", "WECC")
wind$NERC.Region_f <- factor(wind$NERC.Region, levels = levels)
wp_perm_test$NERC.Region_f <- factor(wp_perm_test$NERC.Region, levels = levels)

wind$Wind.Potential <- wind$Wind.Potential / 2000

## Create a wind potential column in the permutation test
## data so that significance can be plotted
wp_perm_test$Wind.Potential <- 1

## Create the ridge plot for wind potential
wind %>%
  ggplot(aes(x = Wind.Potential, y = NERC.Region_f)) +
  geom_density_ridges(
    aes(color = Event.Type, fill = Event.Type),
    quantile_lines = TRUE,
    quantile_fun = median,
    from = 0,
    to = 1,
    scale = 1.1,
    lwd = 0.9,
  ) +
  scale_fill_manual(
    breaks = c("Cold", "Other", "Hot"),
    values = c("NA", "NA", "NA"),
    name = "Event Type",
    guide = "none"
  ) +
  scale_color_manual(
    breaks = c("Cold", "Other", "Hot"),
    values = c("blue", "#0000009A", "red"),
    name = "Event Type",
    guide = "none"
  ) +
  scale_x_continuous(labels=c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(data = wp_perm_test, aes(label = sig), 
            size = 7,
            fontface = "bold",
            color = "firebrick4",
            vjust = -1.25) +
  coord_cartesian(clip = "off") +
  theme_ridges(grid = FALSE) +
  labs(
    x = "Wind Capacity Factor",
    y = "NERC Region",
    # title = "Maximum Wind Potential by Event Type",
    # subtitle = "1980-2021"
  ) +
  facet_wrap(. ~ Season) +
  theme(
    plot.title = element_text(size = 24, face = "bold"),
    title = element_text(size = 20, face = "bold"),
    strip.background = element_blank(),
    axis.title.x = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.y = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 20, face = "bold"),
    strip.text.x = element_text(size = 20, color = "black", face = "bold")
  )

## Save as a PDF
# ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/wp_hollow_ridges_perm_test_capfac.pdf',
#        width = 9,
#        height = 6, units = c("in"))


### Create solar dataframe
solar <- predom_events

## Convert hot and non-extreme days to "other" in DJF
solar$Event.Type[solar$Event.Type=='Hot' &
                  solar$Season=='DJF' | solar$Event.Type=='Non-Extreme'] <- 'Other'

## Convert cold and non-extreme days to "other" in JJA
solar$Event.Type[solar$Event.Type=='Cold' &
                  solar$Season=='JJA' | solar$Event.Type=='Non-Extreme'] <- 'Other'



## Set ordering of NERC regions on plot
## and make sure they are factors in the data so they plot right
levels <- c("RFC", "SERC", "TRE", "NPCC", "MRO", "WECC")
solar$NERC.Region_f <- factor(solar$NERC.Region, levels = levels)
sp_perm_test$NERC.Region_f <- factor(sp_perm_test$NERC.Region, levels = levels)

## Create a solar potential column in the permutation test
## data so that significance can be plotted
sp_perm_test$Solar.Potential <- 0.4

## Create the ridge plot for solar potential
solarplot <- solar %>%
  ggplot(aes(y = NERC.Region_f)) +
  geom_density_ridges(
    aes(x = Solar.Potential, y = NERC.Region_f,
        color = Event.Type, fill = Event.Type),
    quantile_lines = TRUE,
    quantile_fun = median,
    from = 0,
    to = 0.4,
    scale = 1.1,
    lwd = 0.9,
  ) +
  scale_fill_manual(
    breaks = c("Cold", "Other", "Hot"),
    values = c("NA", "NA", "NA"),
    name = "Event Type",
    guide = "none"
  ) +
  scale_color_manual(
    breaks = c("Cold", "Other", "Hot"),
    values = c("blue", "#0000009A", "red"),
    name = "Event Type",
    guide = "none"
  ) +
  scale_x_continuous(labels=c(0, 0.1, 0.2, 0.3, 0.4)) +
  geom_text(data = sp_perm_test, aes(x = Solar.Potential, label = sig),
            size = 7,
            fontface = "bold",
            color = "firebrick4",
            vjust = -1.25) +
  coord_cartesian(clip = "off") +
  theme_ridges(grid = FALSE) +
  labs(
    x = "Solar Capacity Factor",
    y = "",
    # title = "Solar Potential",
    # subtitle = "1980-2021"
  ) +
  facet_wrap(. ~Season) +
  theme(
    plot.title = element_text(size = 24, face = "bold"),
    title = element_text(size = 20, face = "bold"),
    strip.background = element_blank(),
    axis.title.x = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.y = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 20, face = "bold"),
    strip.text.x = element_text(size = 20, color = "black", face = "bold")
  )

## Save as a PDF
# ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/sp_hollow_ridges_perm_test.pdf',
#        width = 9,
#        height = 6, units = c("in"))

## For both capacity factor plots together
plot_grid(windplot, solarplot)

## Save as a PDF
ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/capfac_ridges_perm_test.pdf',
       width = 12,
       height = 6, units = c("in"))


