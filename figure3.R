#### figure3.R
## This script creates the plots used in figure 3
## It loads in weather outage data and the extreme and non-extreme 
## event characteristics file and matches outages to extremes.
## This script outputs the 3 plots used in figure 3
##
##
####

## Import packages
library(ggplot2)
library(lubridate)
library(dplyr)
library(RColorBrewer)
library(tidyverse)
library(purrr)
library(readr)

## Load files
Predom.Event <- read_csv('/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv')
base_dir <-"/Users/yiannabekris/Documents/energy_data/csv/weather_outages/"

## Stack outage files together
all_outages <- list.files(path=base_dir, full.names = TRUE) %>%
  lapply(read_csv) %>%
  bind_rows

## Create month, year, and day columns
all_outages <- all_outages %>%
  mutate(Year = lubridate::year(Date),
                 Month = lubridate::month(Date),
                 Day = lubridate::day(Date)) 

## Change December to year after
all_outages <- all_outages %>% mutate(
  Year = ifelse(
    Month == 12, Year + 1, Year
  )
)

## Remove December 2022 (will actually be December 2021)
## Remove 2011 (aside from December 2011 which is considered 2012 here)
all_outages <- all_outages %>% filter(
  !Year %in% c(2011, 2022)
)


## Add seasons column
all_outages <- all_outages %>% mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

## Find distinct days with outages
all_outages <- all_outages %>% distinct(Date, NERC.Region, .keep_all = TRUE)

## Filter widespread events dataframe by the time period of outage data
Events <- Predom.Event %>% filter(Year %in% 2012:2021)

## Join events and event outages for plotting energy proxy
Event.Outages <- all_outages %>% 
  left_join(Events, by = c(
    "Date", "NERC.Region", "Season","Year", "Month", "Day"
    ))

## Create a copy in order to retain hot and cold categories for energy proxy plots
EnProx.Copy <- Events
EnProx.Outages <- Event.Outages

## Classify hot events in DJF as "Non-Extreme" and cold events in JJA as "Non-Extreme" 
Events$Event.Type[Events$Event.Type=="Hot" & Events$Season=="DJF" |
                    Events$Event.Type=="Cold" & Events$Season=="JJA"] <- "Non-Extreme"

## Classify "Hot" and "Cold" as"Extreme"
Events$Event.Type[Events$Event.Type=="Hot" | Events$Event.Type=="Cold"] <- "Extreme"


## Filter by the seasons and their extreme event types
extreme_events <- Events %>% 
  filter(Event.Type=="Extreme")

## Join extreme dataframe with outage dataframe
outage_join <- left_join(extreme_events, all_outages, 
            by = c(
              "Date","NERC.Region","Year",
                   "Season","Month","Day"
              ))

## Count number of extreme events
Extreme.Counts <- extreme_events %>%
  group_by(Year, Season, NERC.Region) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Season, NERC.Region, fill = list(n = 0))


# Count total number of outages on
outage_totals <- outage_join %>%
  filter(
    Event.ID != "NA"
    ) %>% 
  group_by(NERC.Region, Event.Type, Season, Year,
           .drop = FALSE) %>%
  count() %>%
  ungroup() %>%
  complete(NERC.Region, Event.Type, Season, Year, fill = list(n = 0))

## Count total number of outages on WHCE days in DJF and JJA only
outage_count <- outage_join %>%
  filter(
    !Season %in% c("MAM", "SON")
  ) %>% 
  group_by(NERC.Region, Event.Type, Season, Year,
           .drop = FALSE) %>%
  count() %>%
  ungroup() %>%
  complete(NERC.Region, Event.Type, Season, Year, fill = list(n = 0))


## Rename count column
outage_totals <- outage_totals %>% rename("Outage.Total" = "n")

## Join dataframes to calculate fraction
Outage.Fraction <- outage_totals %>% 
  full_join(Extreme.Counts, 
            by = c("NERC.Region", "Season",
                   "Year"))

## Calculate outage fraction
Outage.Fraction$Fraction <- Outage.Fraction$Outage.Total/Outage.Fraction$n

#write_csv(event_totals, '/Users/yiannabekris/Documents/energy_data/csv/event_totals_1_5_80.csv')

## Make the year a factor for correct plotting
Outage.Fraction$Year <- as.factor(Outage.Fraction$Year)

## Turn NA entries to 0
Outage.Fraction$Fraction[is.na(Outage.Fraction$Fraction)] <- 0

## Calculate the percent of days with outages
Outage.Fraction$Percent <- Outage.Fraction$Fraction * 100

## Get color palettes for year
colourCount = length(unique(Outage.Fraction$Year))
getPalette = colorRampPalette(brewer.pal(10, "Purples"))

## Order of NERC regions in plot
levels <- c("WECC","MRO","NPCC","TRE","SERC","RFC")

## Convert NERC.Region to a factor in each plotting dataframe
Outage.Fraction$NERC.Region_f <- factor(Outage.Fraction$NERC.Region, levels=levels)
EnProx.Copy$NERC.Region_f <- factor(EnProx.Copy$NERC.Region, levels=levels)
EnProx.Outages$NERC.Region_f <- factor(Event.Outages$NERC.Region, levels=levels)

## Boxplots of energy proxy
ggplot(EnProx.Copy %>%
         filter(
           !Season %in% c("MAM", "SON") 
         ), aes(x=NERC.Region_f, y=Energy.Proxy)) +
  geom_boxplot(aes(fill=Event.Type)) + 
  scale_fill_manual(values = c("#7777DD","#CC6666","#CCCCCC"),
                    name = "Event Type") +
  facet_wrap(~Season) +
  labs(x="NERC Region") + ylab('Degree Days') + 
  ggtitle('Energy Proxy 1980-2021') +
  theme_bw(base_size = 18) + 
  theme(strip.text = element_text(size = 30),
        title=element_text(size=24,face="bold")) +
  theme(strip.background=element_rect(fill="black"),
        strip.text=element_text(color="white", face="bold")) +
  geom_point(data=EnProx.Outages %>% 
               filter(
                 Season=="DJF" & Event.Type=="Cold"
               ), 
             aes(x=NERC.Region_f, y=na.omit(Energy.Proxy)),
             shape = 21, color = "black", fill="lightskyblue",stroke=1,
             alpha=0.6, size=3, position = position_nudge(x=-0.19)) +
  geom_point(data=EnProx.Outages %>% 
               filter(
                 Season=="JJA" & Event.Type=="Hot"
               ),
             aes(x=NERC.Region_f, y=na.omit(Energy.Proxy)), 
             shape = 21, color = "black", fill="orangered", stroke=1,
             alpha=0.6, size=3, position = position_nudge(x=-0.19)) +
  geom_point(data=EnProx.Outages %>% 
               filter(
                 Event.Type=="Non-Extreme"
               ),
             aes(x=NERC.Region_f, y=na.omit(Energy.Proxy)), 
             shape = 21, color = "black", fill="ghostwhite", stroke=1,
             alpha=0.6, size=3, position = position_nudge(x=0.19))

## Save 
# ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/enprox_dd_1_5_80.pdf',
#        width = 14,
#        height = 8, units = c("in"))

## Reverse the order of year for the outage plot
year_levs <- (2021:2011)
outage_totals$Year <- factor(outage_totals$Year, levels=year_levs)

## Generate a new palette for year
colourCount = length(unique(outage_count$Year))
palette_original <- colorRampPalette(brewer.pal(11, "PuRd"))(n = 11)

## Reverse the order of the palette
palette_reversed <- palette_original[length(palette_original):1]

## Create a new color palette function using the reversed palette
getPalette <- colorRampPalette(colors = palette_reversed)

## Order of NERC regions for plotting
levels <- c("WECC","MRO","NPCC","TRE","SERC","RFC")

## Change in plot
outage_totals$NERC.Region_f <- factor(outage_totals$NERC.Region, levels=levels)

## Create a bar plot, with the bars divided by year
ggplot(outage_totals, aes(x=NERC.Region_f, y=Outage.Total, fill=Year)) +
  geom_bar(stat="identity", color = "black") +
  scale_fill_manual(values = getPalette(colourCount)) +
  facet_wrap(~Season) +
  labs(x="NERC Region") + 
  ylab('Number of WHCE Days with Outages') + 
  ggtitle('Weather Related Power Outages 2012-2021') +
  theme_bw(base_size = 18) + 
  scale_y_continuous(labels = scales::comma) +
  theme(strip.text = element_text(size = 30),
        title=element_text(size=24,face="bold")) +
  theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) 

## Save 
# ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/WHCE_outagedays_1_5_80.pdf',
#        width = 14,
#        height = 8, units = c("in"))



## Create heatmap of the number of events with outages and display
ggplot(Outage.Fraction, aes(x=NERC.Region_f, y=Year, fill=Percent)) +
  geom_tile(stat="identity", color="black") +
  scale_fill_gradient(low = "lightgoldenrod2", high = "black") +
  geom_text(aes(label = round(Percent, 0)), 
            color="white", fontface = "bold",
            size=6) +
  facet_wrap(~Season) +
  labs(x="NERC Region") + 
  ylab('Percent of WHCE with Outages') + 
  ggtitle('Weather Related Power Outages 2012-2021') +
  theme_bw(base_size=18) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) + 
  theme(
    strip.text = element_text(size = 30),
    title=element_text(size=24,face="bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) 

## Save 
# ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/percentoutageevents_1_5_80.pdf',
#        width = 14,
#        height = 8, units = c("in"))




