# Setup

PheTK installation:


```python
!pip install phetk --upgrade
```


```python
!pip show PheTK | grep Version
```

**Restart the kernal before proceeding**


```python
from phetk.cohort import Cohort
from phetk.phecode import Phecode
from phetk.phewas import PheWAS
from phetk.plot import Plot
```


```python
import os
import subprocess
import numpy as np
import pandas as pd
import polars as pl
from importlib import reload
import seaborn as sns
import matplotlib, matplotlib.pyplot as plt
```

# PheWAS Prep

## Load cohort


```python
import os
cwd = os.getcwd()
print(cwd)
```


```python
df = pd.read_csv('/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/dataframes/df_per_pid_phetk.csv', parse_dates=['date_t0', 'date_tw', 'date_tm', 'date_ts', 'date_ty', 'date_last_ehr'])
```


```python
df_activity_phe = df[["person_id", 
                   "day_steps", "week_steps", "month_steps", "six_steps", "year_steps", 
                   "day_cadence1", "week_cadence1", "month_cadence1", "six_cadence1", "year_cadence1",
                   "day_cadence30", "week_cadence30", "month_cadence30", "six_cadence30", "year_cadence30",
                   "day_dhrps", "week_dhrps", "month_dhrps", "six_dhrps", "year_dhrps",
                   "date_t0", "date_tw", "date_tm", "date_ts", "date_ty",
                   "date_last_ehr", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"]]

df_activity_phe = df_activity_phe.dropna()

variables = ["steps", "cadence1", "cadence30", "dhrps"]
time_windows = ["day", "week", "month", "six", "year"]

# Map each variable to its divisor
divisors = {
    "steps": 1000,
    "cadence1": 10,
    "cadence30": 10,
    "dhrps": 0.01
}

for var in variables:
    for window in time_windows:
        col_name = f"{window}_{var}"
        new_col_name = f"{col_name}_scaled"

        df_activity_phe[new_col_name] = df_activity_phe[col_name] / divisors[var]
```


```python
df_activity_phe.to_csv('/home/jupyter/df_activity_phe.tsv', sep='\t', index=False)
```

## Add covariates


```python
%%time

cohort = Cohort(platform="aou", aou_db_version=8)

cohort.add_covariates(
    sex_at_birth=True,
    drop_nulls=True,
    cohort_file_path="/home/jupyter/df_activity_phe.tsv",
    output_file_path="/home/jupyter/cohort_with_covariates.tsv"
)
```

Add a new variable for control follow up time (time from first fitbit measurement to participant's date of last EHR):


```python
df = pl.read_csv("/home/jupyter/cohort_with_covariates.tsv", separator='\t')
df = df.with_columns([
    pl.col("date_t0").str.strptime(pl.Date, "%Y-%m-%d").alias("date_t0"),
    pl.col("date_last_ehr").str.strptime(pl.Date, "%Y-%m-%d").alias("date_last_ehr")
])

df = df.with_columns(
    (
        (pl.col("date_last_ehr") - pl.col("date_t0"))
        .dt.total_days() / 365.25
    ).alias("control_follow_up")
)
```

Resave as tsv:


```python
df.write_csv("/home/jupyter/cohort_with_covariates.tsv", separator='\t')
```

## Query phecodes


```python
%%time
# instantiate class Phecode and provide some basic information
phecode = Phecode(platform="aou")

# generate phecode profiles/counts
phecode.count_phecode(
    phecode_version="X", 
    output_file_path="/home/jupyter/aou_phecode_counts.tsv"
)
```

Exclude participants with first relevant phecode AFTER start of Fitbit monitoring from cases:


```python
df_date = df[["person_id", "date_t0"]]

phecodes = pl.read_csv("/home/jupyter/aou_phecode_counts.tsv", separator='\t')

phecodes_date = phecodes.join(df_date, on="person_id", how="inner")

DATE_FMT = "%Y-%m-%d"

phecodes_date = (
    phecodes_date
    .with_columns(
        pl.col("first_event_date").str.strptime(pl.Date, DATE_FMT, strict=True).alias("first_event_date"),
    )
)

phecodes_filtered = (phecodes_date.filter(pl.col("date_t0") > pl.col("first_event_date")))

phecodes_filtered.write_csv("/home/jupyter/aou_phecode_counts_prevalent.tsv", separator="\t")
```

Add phecode time to event:


```python
phecode.add_phecode_time_to_event(
    phecode_count_file_path="/home/jupyter/aou_phecode_counts.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates.tsv",
    study_start_date_col="date_t0",
    time_unit="years",
    output_file_path="/home/jupyter/phecode_counts_with_time.tsv"
)
```

# Run PheWAS

## Prevalent cases (logistic regression)


```python
configs = [
# Step count
    {
        "independent_variable_of_interest": "day_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_day_steps.tsv",
        "title": "Day - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_day_steps.pdf"
    },
    {
        "independent_variable_of_interest": "week_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_week_steps.tsv",
        "title": "Week - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_week_steps.pdf"
    },
    {
        "independent_variable_of_interest": "month_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_month_steps.tsv",
        "title": "Month - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_month_steps.pdf"
    },
    {
        "independent_variable_of_interest": "six_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_six_steps.tsv",
        "title": "Six months - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_six_steps.pdf"
    },
    {
        "independent_variable_of_interest": "year_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_year_steps.tsv",
        "title": "Year - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_year_steps.pdf"
    },

# Peak 1-min cadence
    {
        "independent_variable_of_interest": "day_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_day_cadence1.tsv",
        "title": "Day - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_day_cadence1.pdf"
    },
    {
        "independent_variable_of_interest": "week_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_week_cadence1.tsv",
        "title": "Week - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_week_cadence1.pdf"
    },
    {
        "independent_variable_of_interest": "month_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_month_cadence1.tsv",
        "title": "Month - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_month_cadence1.pdf"
    },
    {
        "independent_variable_of_interest": "six_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_six_cadence1.tsv",
        "title": "Six months - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_six_cadence1.pdf"
    },
    {
        "independent_variable_of_interest": "year_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_year_cadence1.tsv",
        "title": "Year - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_year_cadence1.pdf"
    },
    
# Peak 30-min cadence
    {
        "independent_variable_of_interest": "day_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_day_cadence30.tsv",
        "title": "Day - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_day_cadence30.pdf"
    },
    {
        "independent_variable_of_interest": "week_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_week_cadence30.tsv",
        "title": "Week - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_week_cadence30.pdf"
    },
    {
        "independent_variable_of_interest": "month_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_month_cadence30.tsv",
        "title": "Month - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_month_cadence30.pdf"
    },
    {
        "independent_variable_of_interest": "six_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_six_cadence30.tsv",
        "title": "Six months - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_six_cadence30.pdf"
    },
    {
        "independent_variable_of_interest": "year_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_year_cadence30.tsv",
        "title": "Year - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_year_cadence30.pdf"
    },
    
# Daily heart rate per step
    {
        "independent_variable_of_interest": "day_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_day_dhrps.tsv",
        "title": "Day - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_day_dhrps.pdf"
    },
    {
        "independent_variable_of_interest": "week_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_week_dhrps.tsv",
        "title": "Week - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_week_dhrps.pdf"
    },
    {
        "independent_variable_of_interest": "month_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_month_dhrps.tsv",
        "title": "Month - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_month_dhrps.pdf"
    },
    {
        "independent_variable_of_interest": "six_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_six_dhrps.tsv",
        "title": "Six months - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_six_dhrps.pdf"
    },
    {
        "independent_variable_of_interest": "year_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_logistic_year_dhrps.tsv",
        "title": "Year - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_logistic_year_dhrps.pdf"
    }
]

common_params = {
    "phecode_version": "X",
    "phecode_count_file_path": "/home/jupyter/aou_phecode_counts_prevalent.tsv",
    "cohort_file_path": "/home/jupyter/cohort_with_covariates.tsv",
    "sex_at_birth_col": "sex_at_birth",
    "male_as_one": True,
    "covariate_cols": [
        "sex_at_birth",
        "race_eth",
        "alcohol",
        "smoking",
        "income",
        "age_t0",
        "wear_status"
    ],
    "min_cases": 50,
    "min_phecode_count": 2,
    "method": "logit",
}

color_palette = ("#FBE183", "#F4C40F", "#FE9B00", "#F97306", "#D8443C", 
                 "#9B3441", "#DE597C", "#E87B89", "#E6A2A6", "#AA7AA1", 
                 "#9F5691", "#633372", "#3B3B98", "#1F6E9C", "#2B9B81", 
                 "#2E8B57", "#92C051")

# Run analyses in a loop
for config in configs:
    print(f"\nRunning analysis: {config['title']}")
    
    # Instantiate PheWAS object with combined parameters
    phewas = PheWAS(
        **common_params,
        independent_variable_of_interest=config["independent_variable_of_interest"],
        output_file_path=config["output_file_path"]
    )
    
    # Run PheWAS
    phewas.run()
    
    # Generate Manhattan plot
    p = Plot(config["output_file_path"], color_palette=color_palette)
    p.manhattan(
        label_values="p_value",
        label_count=8,
        label_size=15,
        save_plot=True,
        title=config["title"],
        output_file_path=config["plot_output"]
    )
```

## Incident cases (Cox regression)


```python
configs = [
# Step count
    {
        "cox_start_date_col": "date_t0",
        "independent_variable_of_interest": "day_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_day_steps.tsv",
        "title": "Day - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_day_steps.pdf"
    },
    {
        "cox_start_date_col": "date_tw",
        "independent_variable_of_interest": "week_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_week_steps.tsv",
        "title": "Week - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_week_steps.pdf"
    },
    {
        "cox_start_date_col": "date_tm",
        "independent_variable_of_interest": "month_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_month_steps.tsv",
        "title": "Month - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_month_steps.pdf"
    },
    {
        "cox_start_date_col": "date_ts",
        "independent_variable_of_interest": "six_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_six_steps.tsv",
        "title": "Six months - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_six_steps.pdf"
    },
    {
        "cox_start_date_col": "date_ty",
        "independent_variable_of_interest": "year_steps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_year_steps.tsv",
        "title": "Year - steps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_year_steps.pdf"
    },

# Peak 1-min cadence
    {
        "cox_start_date_col": "date_t0",
        "independent_variable_of_interest": "day_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_day_cadence1.tsv",
        "title": "Day - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_day_cadence1.pdf"
    },
    {
        "cox_start_date_col": "date_tw",
        "independent_variable_of_interest": "week_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_week_cadence1.tsv",
        "title": "Week - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_week_cadence1.pdf"
    },
    {
        "cox_start_date_col": "date_tm",
        "independent_variable_of_interest": "month_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_month_cadence1.tsv",
        "title": "Month - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_month_cadence1.pdf"
    },
    {
        "cox_start_date_col": "date_ts",
        "independent_variable_of_interest": "six_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_six_cadence1.tsv",
        "title": "Six months - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_six_cadence1.pdf"
    },
    {
        "cox_start_date_col": "date_ty",
        "independent_variable_of_interest": "year_cadence1_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_year_cadence1.tsv",
        "title": "Year - cadence1",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_year_cadence1.pdf"
    },
    
# Peak 30-min cadence
    {
        "cox_start_date_col": "date_t0",
        "independent_variable_of_interest": "day_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_day_cadence30.tsv",
        "title": "Day - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_day_cadence30.pdf"
    },
    {
        "cox_start_date_col": "date_tw",
        "independent_variable_of_interest": "week_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_week_cadence30.tsv",
        "title": "Week - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_week_cadence30.pdf"
    },
    {
        "cox_start_date_col": "date_tm",
        "independent_variable_of_interest": "month_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_month_cadence30.tsv",
        "title": "Month - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_month_cadence30.pdf"
    },
    {
        "cox_start_date_col": "date_ts",
        "independent_variable_of_interest": "six_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_six_cadence30.tsv",
        "title": "Six months - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_six_cadence30.pdf"
    },
    {
        "cox_start_date_col": "date_ty",
        "independent_variable_of_interest": "year_cadence30_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_year_cadence30.tsv",
        "title": "Year - cadence30",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_year_cadence30.pdf"
    },
    
# Daily heart rate per step
    {
        "cox_start_date_col": "date_t0",
        "independent_variable_of_interest": "day_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_day_dhrps.tsv",
        "title": "Day - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_day_dhrps.pdf"
    },
    {
        "cox_start_date_col": "date_tw",
        "independent_variable_of_interest": "week_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_week_dhrps.tsv",
        "title": "Week - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_week_dhrps.pdf"
    },
    {
        "cox_start_date_col": "date_tm",
        "independent_variable_of_interest": "month_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_month_dhrps.tsv",
        "title": "Month - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_month_dhrps.pdf"
    },
    {
        "cox_start_date_col": "date_ts",
        "independent_variable_of_interest": "six_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_six_dhrps.tsv",
        "title": "Six months - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_six_dhrps.pdf"
    },
    {
        "cox_start_date_col": "date_ty",
        "independent_variable_of_interest": "year_dhrps_scaled",
        "output_file_path": "/home/jupyter/phewas_cox_year_dhrps.tsv",
        "title": "Year - dhrps",
        "plot_output": "/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/manhattan/phewas_cox_year_dhrps.pdf"
    }
]

common_params = {
    "phecode_version": "X",
    "phecode_count_file_path": "/home/jupyter/phecode_counts_with_time.tsv",
    "cohort_file_path": "/home/jupyter/cohort_with_covariates.tsv",
    "sex_at_birth_col": "sex_at_birth",
    "male_as_one": True,
    "cox_control_observed_time_col": "control_follow_up",
    "cox_phecode_observed_time_col": "phecode_time_to_event",
    "covariate_cols": [
        "sex_at_birth",
        "race_eth",
        "alcohol",
        "smoking",
        "income",
        "age_t0",
        "wear_status"
    ],
    "min_cases": 50,
    "min_phecode_count": 2,
    "method": "cox",
    "batch_size": 8
}

color_palette = ("#FBE183", "#F4C40F", "#FE9B00", "#F97306", "#D8443C", 
                 "#9B3441", "#DE597C", "#E87B89", "#E6A2A6", "#AA7AA1", 
                 "#9F5691", "#633372", "#3B3B98", "#1F6E9C", "#2B9B81", 
                 "#2E8B57", "#92C051")

# Run analyses in a loop
for config in configs:
    print(f"\nRunning analysis: {config['title']}")
    
    # Instantiate PheWAS object with combined parameters
    phewas = PheWAS(
        **common_params,
        cox_start_date_col=config["cox_start_date_col"],
        independent_variable_of_interest=config["independent_variable_of_interest"],
        output_file_path=config["output_file_path"]
    )
    
    # Run PheWAS
    phewas.run()
    
    # Generate Manhattan plot
    p = Plot(config["output_file_path"], color_palette=color_palette)
    p.manhattan(
        label_values="p_value",
        label_count=8,
        label_size=15,
        save_plot=True,
        title=config["title"],
        output_file_path=config["plot_output"]
    )
```
