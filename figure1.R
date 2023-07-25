### figure1.R
## This creates the characteristics for figure 1
## First it plots individual plots of each characteristics
## Then it combines them into one plot using cowplot
### ======================================================== ###

## Load packages
library(ggplot2)
library(dplyr)
library(cowplot)
library(lubridate)

## Scale function for labels
scaleFUN <- function(x) sprintf("%.1e", x)

## Read in data
annual <- read.csv('/Users/yiannabekris/Documents/energy_data/csv/annual_metrics_1_5_anom.csv')
Predom.Event <- read_csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")

## Filter to extreme events
Extreme.Events <- Predom.Event %>%
  filter(
    Event.Type != "Non-Extreme"
  )

## Make sure CI is not negative for cold anomalies
Extreme.Events <- Extreme.Events %>%
    mutate(CI = case_when(
      CI > 0 ~ CI,
      CI < 0 ~ CI * -1
    ))

## Frequency calculation
frequency <- Extreme.Events %>%
  group_by(Year, Season, NERC.Region) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Season, NERC.Region, fill = list(n = 0))

## Remove 1980 for DJF as it is added back in through the frequency function
frequency <- frequency %>% 
  filter(!Season %in% c("DJF") | !Year %in% 1980)

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

## Calculate the year with the maximum frequency
frequency_max <- frequency %>% 
  group_by(
    NERC.Region, Season,
  )  %>%
  slice(which.max(n)) %>%
  mutate(labjust = case_when(
  Season=="JJA" ~ 0,
  Season=="DJF" ~ 0.95
) 
)

## Calculate the year with the maximum extent
extent_max <- Extreme.Events %>%
  filter()
  group_by(
    NERC.Region, Season,
  )  %>%
  slice(which.max(Extent)) %>%
  mutate(labjust = case_when(
    Season=="JJA" ~ 0,
    Season=="DJF" ~ 0.95
  ) 
  )

## Calculate the year with the maximum cumulative intensity
ci_max <- Extreme.Events %>% 
  group_by(
    NERC.Region, Season,
  )  %>%
  slice(which.max(CI)) %>%
  mutate(labjust = case_when(
    Season=="JJA" ~ 0,
    Season=="DJF" ~ 0.95
  ) 
  )

## Calculate the year with the duration
duration_max <- duration %>% 
  group_by(
    NERC.Region, Season,
  )  %>%
  slice(which.max(Duration)) %>%
  mutate(labjust = case_when(
    Season=="JJA" ~ 0,
    Season=="DJF" ~ 0.95
  ) 
  )


## Levels for ordering NERC regions correctly
levels <- c("WECC","MRO","NPCC","TRE","SERC","RFC")

## Change NERC region to factor for correct orders
Extreme.Events$NERC.Region_f <- factor(Extreme.Events$NERC.Region, levels = levels)
frequency$NERC.Region_f <- factor(frequency$NERC.Region, levels = levels)
duration$NERC.Region_f <- factor(duration$NERC.Region, levels = levels)
frequency_max$NERC.Region_f <- factor(frequency_max$NERC.Region, levels = levels)
extent_max$NERC.Region_f <- factor(extent_max$NERC.Region, levels = levels)
duration_max$NERC.Region_f <- factor(duration_max$NERC.Region, levels = levels)
ci_max$NERC.Region_f <- factor(ci_max$NERC.Region, levels = levels)

### ============= Individual plots ============= ### 
## Frequency plot
freq_plot <- ggplot(frequency, aes(x=NERC.Region_f, y=n, fill=Season)) +
  geom_boxplot(aes(fill=Season), outlier.shape = NA) +
  scale_fill_manual(values = c("#7777DD","#CC6666"),
                    name = "Season") +
  ylim(0, 60) +
  theme_bw(base_size = 12) + 
  theme(strip.text = element_text(size = 20, color="black", face="bold"),
        title=element_text(size=20,face="bold"),
        legend.position = 'none',
        strip.background=element_rect(fill="white", color = 'white'),
        # axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.x=element_blank(),
        axis.title.x = element_blank()) +
  ylab("Frequency (Days)") + 
  stat_summary(
    aes(fill = Season),
    fun = max,
    geom = "point",
    size = 2,
    position = position_dodge(width = 0.75)) +
  geom_text(
    data=frequency_max, aes(y=n+2, label=Year),
    hjust=frequency_max$labjust,
    position = position_dodge(width = 0.75),
    color='black', size = 4, check_overlap = TRUE)
  #scale_y_discrete(expand = expansion(mult=c(0.1)))

freq_plot

## Set the figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/frequency_boxes_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 7.5,
       height = 6, units = c("in"))

## Extent plot
extent_plot <- ggplot(Extreme.Events, aes(x=NERC.Region_f, y=Extent, fill=Season)) +
  geom_boxplot(aes(fill=Season), outlier.shape = NA) +
  scale_fill_manual(values = c("#7777DD","#CC6666"),
                    name = "Season") +
  theme_bw(base_size = 12) + ylim(0,105) +
  theme(strip.text = element_text(size = 20, color="black", face="bold"),
        title=element_text(size=20,face="bold"),
        legend.position = 'none',
        strip.background=element_rect(fill="white", color = 'white'),
        #axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.x=element_blank(),
        axis.title.x = element_blank()) +
  ylab("Extent %") + 
  stat_summary(
    aes(fill = Season),
    fun = max,
    geom = "point",
    size = 2,
    position = position_dodge(width = 0.75)) +
  geom_text(
    data=extent_max, aes(y=Extent+2.5, label=Year),
    hjust=extent_max$labjust, color='black', 
    position = position_dodge(width = 0.75),
    size = 4, check_overlap = TRUE) 

extent_plot

## Set the figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/extent_boxes_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 7.5,
       height = 6, units = c("in"))

## Duration plot
duration_plot <- ggplot(duration, aes(x=NERC.Region_f, y=Duration, fill=Season)) +
  geom_boxplot(aes(fill=Season), outlier.shape = NA) +
  scale_fill_manual(values = c("#7777DD","#CC6666"),
                    name = "Season") +
  theme_bw(base_size = 12) + 
  theme(strip.text = element_text(size = 20, color="black", face="bold"),
        title=element_text(size=20,face="bold"),
        legend.position = 'none',
        strip.background=element_rect(fill="white", color = 'white'),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank()) +
  ylab("Duration (Days)") + 
  stat_summary(
    aes(fill = Season),
    fun = max,
    geom = "point",
    size = 2,
    position = position_dodge(width = 0.75)) +
  geom_text(
    data=duration_max, aes(y=Duration+0.75, label=Year),
    hjust=duration_max$labjust, color='black', 
    position = position_dodge(width = 0.75),
    size = 4, check_overlap = TRUE) 

duration_plot

## Set the figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/duration_boxes_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 7.5,
       height = 6, units = c("in"))

### CI plot
ci_plot <- ggplot(Extreme.Events, aes(x=NERC.Region_f, y=CI, fill=Season)) +
  geom_boxplot(aes(fill=Season), outlier.shape = NA) +
  scale_fill_manual(values = c("#7777DD","#CC6666"),
                    name = "Season") +
  theme_bw(base_size = 12) + 
  theme(strip.text = element_text(size = 20, color="black", face="bold"),
        title=element_text(size=20,face="bold"),
        legend.position = 'none',
        strip.background=element_rect(fill="white", color = 'white'),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 77),
        axis.title.x=element_blank()) +
  ylab("CI") +
  stat_summary(
    aes(fill = Season),
    fun = max,
    geom = "point",
    size = 2,
    position = position_dodge(width = 0.75)) +
  geom_text(
    data=ci_max, aes(y=CI+500000, label=Year),
    hjust=ci_max$labjust, color='black',
    position = position_dodge(width = 0.75),
    size = 4, check_overlap = TRUE) +
  scale_y_continuous(labels=scaleFUN)

ci_plot 

fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/pdfs/ci_boxes_1_5_80.pdf'

## Save 
ggsave(fig_name,
       width = 7.5,
       height = 6, units = c("in"))


### ============= Plots to 4 panel plot ============= ### 
plot_grid(freq_plot, extent_plot, duration_plot, ci_plot, nrow = 2)

## Set the figure name
fig_name <- '/Users/yiannabekris/Documents/energy_data/figures/final_figures/char_boxes_1_5_80_wide.pdf'

## Save figure
ggsave(fig_name,
       width = 15,
       height = 9, units = c("in"))
  


