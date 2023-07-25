##### mode_perm_tests.R
## This script conducts permutation tests
## on the positive and negative phases of each mode
## in this study (AO, NAO, PNA, El Nino 3.4)
## to assess their relationship to each
## characteristic (duration, extent, frequency, cumulative intensity)
## The output is Figures 5 and S4
##
##### Permutation Test Procedure
## 1) Calculate difference in means
## 2) Combine the 2 distributions
## 3) Sample with replacement from the new distribution (length of dist1 and dist2) (do n times)
## 4) Calculate difference between difference between new dist1 and dist2 (do n times)
## 5) Calculate the 5th and 95th percentile of the differences 
## 6) See if the actual difference is outside of the bounds of the 5th and 95th percentile
##  -- if it is outside of those bounds it means it is significant 
## output of the function should be significant or not significant
#### ===============================================================


### Import packages
library(ggplot2)
library(broom)
library(tidyverse)
library(dplyr)

### Delineate number of repeats
repeats = 1000

### ============= Permutation Test Function ============= ### 
## data is the data to test in 2D form
## column_name is the name of the column
## repeats is the number of samples and difference calculations
run_permutation <- function(data, column_name, repeats){

  ## Find the index of the specified column
  col_index <- which(names(data) == column_name)
  col_expr <- quo(!!sym(column_name))
  
  ## Difference in medians
  median_diff_og <- na.omit(data) %>%
    group_by(NERC.Region, Season, Phase) %>%
    summarise(Median = median(!!col_expr)) %>%
    mutate(Median.Diff = ifelse(length(Median) == 0, 0, diff(Median, 1)))

  ## Filter to just positive as the values of the difference
  ## in medians are the same for positive and negative
  median_diff_filt <- median_diff_og %>% filter(
    Phase=="Positive"
  )

  ## Remove unnecessary columns
  median_diff_filt <- median_diff_filt[,-c(3,4)]
  
  ## Initiate empty tibble to store all differences in medians
  median_diffs <- tibble()
  
  ## Conduct the permutation test and repeat as many times as
  ## delineated by the variables repeats
  for(i in 1:repeats){
     
    ## Remove NA and sum the number of positive and negative observations
    ## then take the median of the column specified for the positive and negative
    ## phases of each mode (grouped by NERC region and Season)
    medians <- na.omit(data) %>%
      group_by(NERC.Region, Season) %>%
      summarise(n_pos = sum(Phase=="Positive"),
                n_neg = sum(Phase=="Negative"),
                pos_median = median(sample((!!col_expr), n_pos, TRUE)),
                neg_median = median(sample((!!col_expr), n_neg, TRUE)))
    
    ## Find the difference of medians
    median_diff <- medians %>%
      group_by(NERC.Region, Season) %>%
      summarise(median_diff = sum(pos_median - neg_median))
    
    ## Bind to tibble of differences in medians
    median_diffs <- rbind(median_diffs, median_diff)
    
  }
  
  ## Find the 5th and 95th percentile of the difference in medians
  quants_diff <- median_diffs %>%
    group_by(NERC.Region, Season) %>%
    summarise(quant_5 = quantile(median_diff, probs = 0.05, na.rm = TRUE),
              quant_95 = quantile(median_diff, probs = 0.95, na.rm = TRUE))
  
  ## Join the original difference in medians to the 5th and 95th percentile dataframe
  median_quants_diff <- full_join(quants_diff, median_diff_filt,
                                  by = c("NERC.Region", "Season"))
  
  ## Create a column which shows significance if the median difference in medians
  ## is less than the 5th percentile or greater than the 95th percentile
  quants_mutate <- median_quants_diff %>% 
    mutate(
      sig = ifelse(Median.Diff < quant_5 | Median.Diff > quant_95, "*", "")
    )

  ## Add in a column of the mode for plotting  
  quants_mutate$Mode <- data$Mode[1]
  
  return(quants_mutate)
}  



### ============= Preparing data for plotting ============= ### 
## Read in file with event data
Predom.Event <- read.csv("/Users/yiannabekris/Documents/energy_data/csv/predom_event_1_5_80.csv")

## Load files with mode data (+/- 0.5 is the threshold for positive and negative phases)
ao <- read_csv("/Users/yiannabekris/Documents/energy_data/clim_indices/ao_0_5_thresh.csv")
nao <- read_csv("/Users/yiannabekris/Documents/energy_data/clim_indices/nao_0_5_thresh.csv")
nino34 <- read_csv("/Users/yiannabekris/Documents/energy_data/clim_indices/nino34_0_5_thresh.csv")
pna <- read_csv("/Users/yiannabekris/Documents/energy_data/clim_indices/pna_0_5_thresh.csv")

## Remove neutral-phase observations
ao <- ao %>% filter(Phase != "Neutral")
nao <- nao %>% filter(Phase != "Neutral")
nino34 <- nino34 %>% filter(Phase != "Neutral")
pna <- pna %>% filter(Phase != "Neutral")

## Filter event data to only extreme events
Extreme.Events <- Predom.Event %>%
  filter(
    Event.Type != "Non-Extreme"
  )


### ============= Permutation tests ============= ###
## Join each mode dataframe to the event dataframe
events_ao <- left_join(Extreme.Events, ao, 
                       by = c("Month", "Year", "Season"))

events_nao <- left_join(Extreme.Events, nao,
                        by = c("Month", "Year", "Season"))

events_nino34 <- left_join(Extreme.Events, nino34,
                           by = c("Month", "Year", "Season"))

events_pna <- left_join(Extreme.Events, pna,
                        by = c("Month", "Year", "Season"))

## Bind all together for plotting
events_all_modes <- rbind(events_ao, events_nao, events_nino34, events_pna)

### ============= Cumulative Intensity Tests ============= ### 
## Create a list with data for all events and modes
ci_modes_list <- list(events_ao, events_nao, events_nino34, events_pna)

## Initiate an empty tibble to store results
ci_perm <- tibble()

## Loop through and conduct permutation test
for(mode in ci_modes_list){
  
  ## Remove NAs
  mode <- na.omit(mode)
  
  ## Find # of positive and negative observations
  ## If there are less then 20 observations for both positive and negative
  ## phases, then the permutation test will not be run
  mode_test <- mode %>%
    group_by(NERC.Region, Season) %>%
    mutate(n_pos = sum(Phase == "Positive"),
           n_neg = sum(Phase == "Negative"))

  
  ## Conduct permutation tests
  if(any(mode_test$n_pos >= 20 & mode_test$n_neg >=20)) {
    
    ## Filter out observations if there are not 20 instances of
    ## positive or negative phases
    mode_filt <- mode_test %>%
      filter((n_pos >= 20) & (n_neg >= 20)) %>%
      ungroup()

    ## Run the permutation test on the above filtered data
    pos_neg <- run_permutation(mode_filt, "CI", repeats)

    ## Bind to list for plotting
    ci_perm <- rbind(ci_perm, pos_neg)
 
  }  
}
  

### ============= Plotting preparation ============= ###
## Set the order of NERC regions
levels <- c("WECC","MRO","NPCC","TRE","SERC","RFC")

## Convert to factor in a new column
ci_perm$NERC.Region_f <- factor(ci_perm$NERC.Region, levels=levels)
events_all_modes$NERC.Region_f <- factor(events_all_modes$NERC.Region, levels=levels)

## Join the permutation test results to the mode dataframe
ci_plot <- left_join(
  events_all_modes, ci_perm, 
    by = c(
  "Season", "Mode", "NERC.Region", "NERC.Region_f")
)

## Remove NAs
ci_plot <- ci_plot %>% 
  filter(!is.na(Mode))

## Calculate the number of positive and negative observations
ci_plot <- ci_plot %>%
  group_by(NERC.Region, Season, Mode) %>%
  mutate(n_pos = sum(Phase == "Positive"),
         n_neg = sum(Phase == "Negative"))

## Only plot if there are 20 or more observations for both negative and positive phases
## for each NERC region and season
ci_plot <- ci_plot %>%
  group_by(NERC.Region, Season) %>%
  filter((n_pos >= 20) & (n_neg >= 20)) %>%
  ungroup()


### ============= Cumulative Intensity Boxplots ============= ### 
ggplot(ci_plot, aes(x=Mode, y=CI)) +
  geom_boxplot(aes(fill=Phase), outlier.shape = NA) + 
  scale_fill_manual(values = c("#FFCC00","#9966FF"),
                    name = "Phase") +
  facet_grid(
    rows = vars(Season),
    cols = vars(NERC.Region_f),
    scales = "free_y"
  ) + 
  scale_y_continuous(
    limits = ifelse(ci_plot$Season == "DJF", c(0, 5000000), 
                    ifelse(ci_plot$Season == "JJA", c(0, 3000000), NA))
  ) + geom_text(
    data = ci_perm, 
    aes(x = Mode, y = 3000000, label = sig), 
    vjust = 1,
    # check_overlap = TRUE,
    position = position_dodge(width = 0.75),
    ) + 
  labs(x="Mode") + ylab('CI') + ggtitle('Widespread Event CI 1980-2021') +
  theme_bw() + theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold"),
        axis.text.x = element_text(angle = 45, hjust = 1)) + 
  theme(strip.text = element_text(size = 30),
        title=element_text(size=24,face="bold")) 

## Save 
ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/ci_05modes_perm_1_5_80_1000.pdf',
       width = 14,
       height = 8, units = c("in"))

### ============= Extent Tests ============= ###
## Create a list with data for all events and modes
extent_modes_list <- list(events_ao, events_nao, events_nino34, events_pna)

## Initiate an empty tibble to store results
extent_perm <- tibble()

## Loop through and conduct permutation test
for(mode in extent_modes_list){
  
  ## Remove NAs
  mode <- na.omit(mode)

  ## Find # of positive and negative observations
  ## If there are less then 20 observations for both positive and negative
  ## phases, then the permutation test will not be run
  mode_test_extent <- mode %>%
    group_by(NERC.Region, Season) %>%
    mutate(n_pos = sum(Phase == "Positive"),
           n_neg = sum(Phase == "Negative"))

  
  ## Conduct permutation tests   
  if(any(mode_test$n_pos >= 20 & mode_test$n_neg >=20)) {
    
    ## Filter out observations if there are not 20 instances of
    ## positive or negative phases
    mode_filt_extent <- mode_test_extent %>%
      filter((n_pos >= 20) & (n_neg >= 20)) %>%
      ungroup()

    ## Run the permutation test on the above filtered data
    pos_neg_extent <- run_permutation(mode_filt_extent, "Extent", repeats)
    
    ## Bind to list for plotting
    extent_perm <- rbind(extent_perm, pos_neg_extent)
    
  }
  
}


### ============= Plotting preparation ============= ###
## Set the order of NERC regions
extent_perm$NERC.Region_f <- factor(extent_perm$NERC.Region, levels=levels)

## Join the permutation test results to the mode dataframe
extent_plot <- left_join(
  events_all_modes, extent_perm, 
  by = c(
    "Season", "Mode", "NERC.Region", "NERC.Region_f")
)


## Calculate the number of positive and negative observations
extent_plot <- extent_plot %>%
  group_by(NERC.Region, Season, Mode) %>%
  mutate(n_pos = sum(Phase == "Positive"),
         n_neg = sum(Phase == "Negative"))


## Only plot if there are 20 or more observations for both negative and positive phases
## for each NERC region and season
extent_plot <- extent_plot %>% 
  group_by(NERC.Region, Season, Mode) %>%
  filter(!is.na(Mode) &
           ((n_pos >= 20) & (n_neg >= 20))) %>%
           ungroup()

### ============= Extent Boxplots ============= ### 
ggplot(extent_plot, aes(x=Mode, y=Extent)) +
  geom_boxplot(aes(fill=Phase), outlier.shape = NA) + 
  scale_fill_manual(values = c("#FFCC00","#9966FF"),
                    name = "Phase") +
  facet_grid(
    rows = vars(Season),
    cols = vars(NERC.Region_f), scales = "free_y"
  ) + geom_text(
    data = extent_perm, 
    aes(x=Mode, y=104, label = sig), 
    vjust = 1
    ) + 
  labs(x="Mode") + ylab('Extent') + 
  ggtitle('Widespread Event Extent 1980-2021') +
  theme_bw() + theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold", size = 30),
        axis.text.x = element_text(angle = 45, hjust = 1),
        title=element_text(size = 24, face = "bold"))

## Save 
ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/extent_05modes_perm_1_5_80_1000.pdf',
       width = 14,
       height = 8, units = c("in"))

### ============= Frequency Tests ============= ###
## Calculate monthly frequency
frequency_monthly <- Extreme.Events %>%
  group_by(Year, Month, NERC.Region) %>%
  count() %>%
  ungroup() %>%
  complete(Year, Month, NERC.Region, fill = list(n = 0))

## Remove DJF 1980 as it will be filled with an artifical 0
## from the complete function above
frequency_monthly <- frequency_monthly %>%
  filter(
    !Month %in% c(1, 2) | !Year %in% 1980
  )

## Add seasons column
frequency_monthly <- frequency_monthly %>% 
  mutate(Season = case_when(
  Month %in% c(6, 7, 8)  ~ "JJA" ,
  Month %in% c(9, 10, 11)  ~ "SON"  ,
  Month %in% c(1, 2, 12)  ~ "DJF"  ,
  Month %in% c(3, 4, 5) ~ "MAM"
)
)

## Create dataframes with indices
frequency_ao <- left_join(frequency_monthly, ao, 
                       by = c("Month", "Year", "Season"))

frequency_nao <- left_join(frequency_monthly, nao,
                        by = c("Month", "Year", "Season"))

frequency_nino34 <- left_join(frequency_monthly, nino34,
                           by = c("Month", "Year", "Season"))

frequency_pna <- left_join(frequency_monthly, pna,
                        by = c("Month", "Year", "Season"))

## Create a dataframe with all modes for plotting
freq_all_modes <- rbind(frequency_ao, frequency_nao, frequency_nino34, frequency_pna)

## Create a list with data for all events and modes
freq_modes_list <- list(frequency_ao, frequency_nao, frequency_nino34, frequency_pna)

## Initiate an empty tibble to store results
frequency_perm <- tibble()

## Loop through and conduct permutation test
for(mode in freq_modes_list){
  
  ## Remove NAs
  mode <- na.omit(mode)

  ## Find # of positive and negative observations
  ## If there are less then 20 observations for both positive and negative
  ## phases, then the permutation test will not be run
  mode_test_freq <- mode %>%
    group_by(NERC.Region, Season) %>%
    mutate(n_pos = sum(Phase == "Positive" & n !=0),
           n_neg = sum(Phase == "Negative" & n !=0))

  ## Conduct permutation tests   
  if(any(mode_test_freq$n_pos >= 20 & mode_test_freq$n_neg >=20)) {
    
    ## Filter out observations if there are not 20 instances of
    ## positive or negative phases
    mode_filt_freq <- mode_test_freq %>%
      filter((n_pos >= 20) & (n_neg >= 20)) %>%
      ungroup()
    
    ## Run the permutation test on the above filtered data
    pos_neg_freq <- run_permutation(mode_filt_freq, "n", repeats)

    ## Bind to list for plotting
    frequency_perm <- rbind(frequency_perm, pos_neg_freq)
    
  }
  
}


### ============= Plotting preparation ============= ###
## Set the order of NERC regions
freq_all_modes$NERC.Region_f <- factor(freq_all_modes$NERC.Region, levels=levels)
frequency_perm$NERC.Region_f <- factor(frequency_perm$NERC.Region, levels=levels)

## Remove NAs
freq_all_modes <- freq_all_modes %>% 
  filter(!is.na(Mode))

## Sum positive and negative phases, but don't include
## months with 0 events
freq_all_modes <- freq_all_modes %>%
  group_by(NERC.Region, Season, Mode) %>%
  mutate(n_pos = sum(Phase == "Positive" & n!=0),
         n_neg = sum(Phase == "Negative" & n!=0))

## Only plot if there are 20 or more observations for both negative and positive phases
## for each NERC region and season
freq_all_modes <- freq_all_modes %>%
  filter((n_pos >= 20) & (n_neg >= 20)) %>%
  ungroup()

### ============= Frequency Boxplots ============= ### 
ggplot(freq_all_modes, aes(x=Mode, y=n)) +
  geom_boxplot(aes(fill=Phase), outlier.shape = NA) + 
  scale_fill_manual(values = c("#FFCC00","#9966FF"),
                    name = "Phase") +
  facet_grid(
    rows = vars(Season),
    cols = vars(NERC.Region_f)
  ) + geom_text(data = frequency_perm, aes(y=25, label = sig), vjust = 1) + 
  labs(x="Mode") + ylab('Frequency (Days)') + ggtitle('Widespread Event Monthly Frequency 1980-2021') +
  theme_bw() + theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) + 
  theme(strip.text = element_text(size = 30),
        axis.text.x = element_text(angle = 45, hjust = 1),
        title=element_text(size=24,face="bold")) 

## Save 
ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/freq_05modes_perm_1_5_80_1000.pdf',
       width = 14,
       height = 8, units = c("in"))


### ============= Duration ============= ### 
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

## Extract month from the date
duration <- duration %>%
  dplyr:: mutate(Month = lubridate::month(Start.Date))

### Join with mode dataframes for the permutation test and plotting
duration_ao <- duration %>% left_join(ao, by=c("Year","Month", "Season"))
duration_nao <- duration %>% left_join(nao, by=c("Year","Month", "Season"))
duration_nino34 <- duration %>% left_join(nino34, by=c("Year","Month", "Season"))
duration_pna <- duration %>% left_join(pna, by=c("Year","Month", "Season"))

## Create a dataframe with all modes for plotting
dur_ind_df <- rbind(duration_ao, duration_nao, duration_nino34, duration_pna)

## Create a list with data for all events and modes
duration_modes_list <- list(duration_ao, duration_nao, duration_nino34, duration_pna)

## Initiate an empty tibble to store results
duration_perm <- tibble()

## Loop through and conduct permutation test
for(mode in duration_modes_list){
  mode <- na.omit(mode)

  ## Find # of positive and negative observations
  mode_test_dur <- mode %>%
    group_by(NERC.Region, Season) %>%
    mutate(n_pos = sum(Phase == "Positive"),
           n_neg = sum(Phase == "Negative"))

  ## Conduct permutation tests   
  if(any(mode_test$n_pos >= 20 & mode_test$n_neg >=20)) {
    
    mode_filt_dur <- mode_test_dur %>%
      # group_by(NERC.Region, Season) %>%
      filter((n_pos >= 20) & (n_neg >= 20)) %>%
      ungroup()
    
    pos_neg_dur <- run_permutation(mode_filt_dur, "Duration", repeats)
    
    ## Bind to list for plotting
    duration_perm <- rbind(duration_perm, pos_neg_dur)
    
  }
  
}
  
### ============= Plotting preparation ============= ###
## Set the order of NERC regions
duration_perm$NERC.Region_f = factor(duration_perm$NERC.Region,levels=levels) 
dur_ind_df$NERC.Region_f = factor(dur_ind_df$NERC.Region,levels=levels) 

## Sum positive and negative phases, but don't include
## months with 0 events
dur_ind_df <- dur_ind_df %>% 
  group_by(NERC.Region, Season, Mode) %>%
  mutate(n_pos = sum(Phase == "Positive"),
         n_neg = sum(Phase == "Negative"))

## Only plot if there are 20 or more observations for both negative and positive phases
## for each NERC region and season
dur_ind_df <- dur_ind_df %>% 
  filter(!is.na(Mode) &
           (n_pos >= 20) & (n_neg >= 20)) %>%
           ungroup()

### ============= Duration Boxplots ============= ### 
ggplot(dur_ind_df, aes(x=Mode, y=Duration)) +
  geom_boxplot(aes(fill=Phase), outlier.shape = NA) + 
  scale_fill_manual(values = c("#FFCC00","#9966FF"),
                    name = "Phase") +
  facet_grid(
    rows = vars(Season),
    cols = vars(NERC.Region_f)
  ) + geom_text(
    data = duration_perm, 
    aes(x=Mode, y=25, label = sig), 
    vjust = 1
  ) + 
  labs(x="Mode") + ylab('Duration (Days)') + ggtitle('Widespread Event Duration 1980-2021') +
  theme_bw() + theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) + 
  theme(strip.text = element_text(size = 30),
        axis.text.x = element_text(angle = 45, hjust = 1),
        title=element_text(size=24,face="bold")) 

## Save 
ggsave('/Users/yiannabekris/Documents/energy_data/figures/pdfs/duration_05modes_perm_1_5_80_1000.pdf',
       width = 14,
       height = 8, units = c("in"))
