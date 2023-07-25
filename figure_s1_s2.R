### figure_s1_s2.R
## This takes the daily characteristics data
## and plots the time series for
## frequency, mean extent, mean cumulative intensity, and mean duration
## as well as annotates their trends over time (1980-2021) and p-values
## The output is figures S1 and S2
###


## Import packages
library(ggplot2)
library(tidyverse)
library(dplyr)


## Read in data
annual <- read.csv('/Users/yiannabekris/Documents/energy_data/csv/annual_metrics_1_5_anom.csv')
Predom.Event <- read.csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")

## Filter to just extreme events
Extreme.Events <- Predom.Event %>%
  filter(
    Event.Type != "Non-Extreme"
  )


## Calculate duration (number of consecutive days of extreme temperature events)
duration_df <- Extreme.Events %>%
  mutate(Date = as.Date(Date)) %>%
  group_by(NERC.Region, Season, Year) %>%
  mutate(Event.ID = cumsum(c(1, diff(Date) > 1))) %>%
  group_by(NERC.Region, Season, Year, Event.ID) %>%
  summarize(Duration = as.numeric(difftime(max(Date), min(Date), 
                                           units = "days")) + 1,
            Start.Date = min(Date),
            .groups = "drop") %>%
  select(-Event.ID)

## Rename Start.Date as Date
duration_df <- duration_df %>% rename("Date" = "Start.Date")

## Add Year column
duration_df <- duration_df %>%
  mutate(
    Year = lubridate::year(Date),
  ) 

## Calculate average duration
duration_df <- aggregate(Duration ~ NERC.Region + Season + Year, 
                          data=duration_df, FUN=mean, na.action = na.omit)

## Pivot longer to bind with other characteristics
duration_long <- duration_df %>% 
  pivot_longer(
    cols = Duration,
    names_to = 'Metric',
    values_to = 'Quantity',
    values_drop_na = TRUE
  )

## Order of columns
col_order <- c("NERC.Region", "Season", "Metric", "Quantity", "Year")

## Average event extent
extent_df <- aggregate(Extent ~ NERC.Region + Season + Year, 
                               data=Extreme.Events, FUN=mean, na.action = na.omit)

## Pivot longer to bind with other characteristics
extent_long <- extent_df %>%
  pivot_longer(
    cols = Extent,
    names_to = 'Metric',
    values_to = 'Quantity',
    values_drop_na = TRUE
  )

## Average event cumulative intensity
ci_df <- aggregate(CI ~ NERC.Region + Season + Year, 
                          data=Extreme.Events, FUN=mean, na.action = na.omit)

## Pivot longer to bind with other characteristics
ci_long <- ci_df %>%
  pivot_longer(
    cols = CI,
    names_to = 'Metric',
    values_to = 'Quantity',
    values_drop_na = TRUE
  )

## Find the annual frequency of extreme events, grouped by NERC region and year
frequency_df <- Extreme.Events %>%
  group_by(
    Year, Season, NERC.Region,
    ) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Season, NERC.Region, fill = list(n = 0))

## Remove DJF 1980 as it will be an artificial 0 count
frequency_long <- frequency_df %>%
  filter(!Season %in% c("DJF") | !Year %in% 1980)

## Add a "Metric" column
frequency_long$Metric <- 'Frequency'

## Change the frequency count to "Quantity"
frequency_long <- frequency_long %>% 
  rename(
    "Quantity" = "n",
  )

## Ensure columns in dataframes are in the correct order for binding
extent <- extent_long[, col_order]
ci <- ci_long[, col_order]
frequency <- frequency_long[, col_order]
duration <- duration_long[, col_order]

characteristics <- rbind(frequency, extent, ci, duration)

## Levels for ordering NERC regions and metrics correctly
levels <- c("WECC","MRO","NPCC","TRE","SERC","RFC")
metric_levels <- c("Frequency", "Extent","Duration", "CI")


## Run linear models for annotations
linear_mods <- characteristics %>%
  group_by(NERC.Region, Season, Metric) %>%
  group_modify(
    ~ tidy(lm(Quantity ~ Year, data = .))
  ) %>% ungroup()

## "Year" contains the coefficient
linear_mods <- linear_mods %>%
  filter(term == "Year")

## Convert NERC region and Metric to factors for correct ordering of facets
characteristics$NERC.Region_f <- factor(characteristics$NERC.Region, levels=levels)
characteristics$Metric_f <- factor(characteristics$Metric, levels=metric_levels)
linear_mods$NERC.Region_f <- factor(linear_mods$NERC.Region, levels=levels)
linear_mods$Metric_f <- factor(linear_mods$Metric, levels=metric_levels)

## Round to 2 significant digits
linear_mods <- linear_mods %>% mutate_if(is.numeric, ~ round(., 2))

## Subset to the labels for each season
djf_labs <- linear_mods %>%
  filter(Season == "DJF")

jja_labs <- linear_mods %>%
  filter(Season == "JJA")

## Filter to just DJF and JJA for plotting
char_djf <- characteristics %>%
  filter(Season == "DJF")

char_jja <- characteristics %>%
  filter(Season == "JJA")

## Set the title
title = 'Widespread Event Characteristics 1980-2021'

## Heatwave dates
heatwave <- Predom.Event %>% 
  filter(
    Event.Type == "Hot" & 
      NERC.Region %in% c("WECC") &
        Date %in% c("2021-06-27","2021-06-28",
                    "2021-06-29","2021-06-30")
    )

### Winter and the coldwave
coldwave <- Predom.Event %>% 
  filter(
    Event.Type == "Cold" &
      NERC.Region %in% c("TRE") &
        Date %in% c("2021-02-12","2021-02-13",
                    "2021-02-14","2021-02-15",
                    "2021-02-16","2021-02-17",
                    "2021-02-18","2021-02-19")
    )

## Pivot the heatwave and coldwave dataframes longer for plotting
heatwave_long <- heatwave %>% 
  pivot_longer(
    cols = c('CI', 'Extent'),
    names_to = 'Metric',
    values_to = 'Quantity'
  )

coldwave_long <- coldwave %>% 
  pivot_longer(
    cols = c('CI', 'Extent'),
    names_to = 'Metric',
    values_to = 'Quantity'
  )

coldwave_long <- coldwave_long[, c("NERC.Region", "Event.Type", "Season", "Year", "Date", "Metric", "Quantity")]

## Convert NERC region and Metric to factors for correct ordering of facets
coldwave_long$NERC.Region_f <- factor(coldwave_long$NERC.Region, levels=levels)
heatwave_long$NERC.Region_f <- factor(heatwave_long$NERC.Region, levels=levels)
coldwave_long$Metric_f <- factor(coldwave_long$Metric, levels=metric_levels)
heatwave_long$Metric_f <- factor(heatwave_long$Metric, levels=metric_levels)

## Cold event time series plots
ggplot(data = char_djf, aes(x = Year, y = Quantity))+
  geom_line(color = "royalblue", size = 0.75) + 
  geom_jitter(data=coldwave_long, aes(x = Year, y = Quantity, fill = Date), shape = 21, 
              color = "black", size=2, alpha=0.5) +
  scale_fill_brewer() + theme_bw() + 
  geom_smooth(method=lm, se=FALSE, size=0.6) + 
  facet_grid(rows = vars(Metric_f),
             cols = vars(NERC.Region_f), scales="free") +
  scale_fill_brewer() + theme_bw() + 
  geom_smooth(method=lm, se=FALSE, size=0.6) +
  theme_bw(base_size = 15) + 
  theme(
    strip.background=element_blank(),
    strip.text=element_text(color="black", face="bold", size = 15),
    title=element_text(size=20,face="bold"),
    axis.text = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
  ) +
  geom_smooth(method=lm, se=FALSE, size=0.8, color="royalblue4") + 
  geom_point(color="black", fill="darkblue", shape=21, alpha=0.8, size=1) +
  geom_text(size=3, data = djf_labs, check_overlap = TRUE,
            mapping = aes(x=Inf, y=Inf, 
                          hjust=1.1, vjust=1.2,
                          label = paste('  Trend = ', estimate, '\n',
                                        'p-value = ', p.value))) +
  ylab('') +
  ggtitle("Cold Extreme Trends DJF 1980-2021") 

## Set cold event figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/all_trends_djf_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 12,
       height = 7, units = c("in"))

## Hot event time series plots
ggplot(data = char_jja, aes(x = Year, y = Quantity))+
  geom_line(color = "red3", size = 0.75) + 
  geom_jitter(data=heatwave_long, aes(x = Year, y = Quantity, fill = Date), shape = 21, 
              color = "black", size=2, alpha=0.5) +
  scale_fill_brewer() + theme_bw() + 
  geom_smooth(method=lm, se=FALSE, size=0.6) + 
  facet_grid(rows = vars(Metric_f),
             cols = vars(NERC.Region_f), scales="free") +
  scale_fill_viridis_d(option="inferno") + theme_bw() + 
  geom_smooth(method=lm, se=FALSE, size=0.6) +
  theme_bw(base_size = 15) + 
  theme(
    strip.background=element_blank(),
    strip.text=element_text(color="black", face="bold", size = 15),
    title=element_text(size=20,face="bold"),
    axis.text = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
  ) +
  geom_smooth(method=lm, se=FALSE, size=0.8, color="darkred") + 
  geom_point(color="black", fill="indianred4", shape=21, alpha=0.8, size=1) +
  geom_text(size=3, data = jja_labs, check_overlap = TRUE,
            mapping = aes(x=-Inf, y=Inf, 
                          hjust=-0.01, vjust=1.2, 
                          label = paste('  Trend = ', estimate, '\n',
                                        'p-value = ', p.value))) +
  ylab('') +
  ggtitle("Hot Extreme Trends JJA 1980-2021") 

## Set hot event figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/all_trends_jja_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 12,
       height = 7, units = c("in"))

