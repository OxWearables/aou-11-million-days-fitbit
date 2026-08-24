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
df = pd.read_csv('/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/dataframes/df_per_pid_phetk.csv', parse_dates=['date_t0', 'date_ty', 'date_t0_washout', 'date_last_ehr'])
```


```python
df.columns
```


```python
df_activity_phe = df[["person_id",
                      "fitbit_year",
                      "year_steps",
                      "year_cadence1",
                      "year_cadence30",
                      "year_dhrps",
                      "date_t0", "date_ty",
                      "date_last_ehr", 
                      "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"]]

df_activity_phe = df_activity_phe.dropna()

variables = ["steps", "cadence1", "cadence30", "dhrps"]
time_windows = ["year"]

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
df_activity_phe_washout = df[["person_id",
                              "fitbit_year",
                              "year_steps_washout",
                              "year_cadence1_washout",
                              "year_cadence30_washout",
                              "year_dhrps_washout",
                              "date_t0_washout",
                              "date_last_ehr", 
                              "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"]]

df_activity_phe_washout = df_activity_phe_washout.dropna()

variables = ["steps_washout", "cadence1_washout", "cadence30_washout", "dhrps_washout"]
time_windows = ["year"]

# Map each variable to its divisor
divisors = {
    "steps_washout": 1000,
    "cadence1_washout": 10,
    "cadence30_washout": 10,
    "dhrps_washout": 0.01
}

for var in variables:
    for window in time_windows:
        col_name = f"{window}_{var}"
        new_col_name = f"{col_name}_scaled"

        df_activity_phe_washout[new_col_name] = df_activity_phe_washout[col_name] / divisors[var]
```


```python
# BMI
df_activity_phe.to_csv('/home/jupyter/df_activity_phe_bmi.tsv', sep='\t', index=False)

# Washout
df_activity_phe_washout.to_csv('/home/jupyter/df_activity_phe_washout.tsv', sep='\t', index=False)

# One year Fitbit
df_activity_phe_one = df_activity_phe[df_activity_phe['fitbit_year'] == 1]
df_activity_phe_one.to_csv('/home/jupyter/df_activity_phe_one.tsv', sep='\t', index=False)

# Exclude first three years
df_activity_phe["date_ty3"] = df_activity_phe["date_ty"] + pd.Timedelta(days=730)
df_activity_phe.to_csv('/home/jupyter/df_activity_phe_three.tsv', sep='\t', index=False)
```

## Add covariates

### BMI


```python
%%time

cohort = Cohort(platform="aou", aou_db_version=8)

cohort.add_covariates(
    sex_at_birth=True,
    drop_nulls=True,
    cohort_file_path="/home/jupyter/df_activity_phe_bmi.tsv",
    output_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv"
)
```


```python
df_bmi = pl.read_csv("/home/jupyter/cohort_with_covariates_bmi.tsv", separator='\t')
df_bmi = df_bmi.with_columns([
    pl.col("date_t0").str.strptime(pl.Date, "%Y-%m-%d").alias("date_t0"),
    pl.col("date_last_ehr").str.strptime(pl.Date, "%Y-%m-%d").alias("date_last_ehr")
])

df_bmi = df_bmi.with_columns(
    (
        (pl.col("date_last_ehr") - pl.col("date_t0"))
        .dt.total_days() / 365.25
    ).alias("control_follow_up")
)
```


```python
df_bmi.write_csv("/home/jupyter/cohort_with_covariates_bmi.tsv", separator='\t')
```

### Washout


```python
%%time

cohort = Cohort(platform="aou", aou_db_version=8)

cohort.add_covariates(
    sex_at_birth=True,
    drop_nulls=True,
    cohort_file_path="/home/jupyter/df_activity_phe_washout.tsv",
    output_file_path="/home/jupyter/cohort_with_covariates_washout.tsv"
)
```


```python
df_washout = pl.read_csv("/home/jupyter/cohort_with_covariates_washout.tsv", separator='\t')
df_washout = df_washout.with_columns([
    pl.col("date_t0_washout").str.strptime(pl.Date, "%Y-%m-%d").alias("date_t0_washout"),
    pl.col("date_last_ehr").str.strptime(pl.Date, "%Y-%m-%d").alias("date_last_ehr")
])

df_washout = df_washout.with_columns(
    (
        (pl.col("date_last_ehr") - pl.col("date_t0_washout"))
        .dt.total_days() / 365.25
    ).alias("control_follow_up")
)
```


```python
df_washout.write_csv("/home/jupyter/cohort_with_covariates_washout.tsv", separator='\t')
```

### One year


```python
%%time

cohort = Cohort(platform="aou", aou_db_version=8)

cohort.add_covariates(
    sex_at_birth=True,
    drop_nulls=True,
    cohort_file_path="/home/jupyter/df_activity_phe_one.tsv",
    output_file_path="/home/jupyter/cohort_with_covariates_one.tsv"
)
```


```python
df_one = pl.read_csv("/home/jupyter/cohort_with_covariates_one.tsv", separator='\t')
df_one = df_one.with_columns([
    pl.col("date_t0").str.strptime(pl.Date, "%Y-%m-%d").alias("date_t0"),
    pl.col("date_last_ehr").str.strptime(pl.Date, "%Y-%m-%d").alias("date_last_ehr")
])

df_one = df_one.with_columns(
    (
        (pl.col("date_last_ehr") - pl.col("date_t0"))
        .dt.total_days() / 365.25
    ).alias("control_follow_up")
)
```


```python
df_one.write_csv("/home/jupyter/cohort_with_covariates_one.tsv", separator='\t')
```

### Exclude first three years


```python
%%time

cohort = Cohort(platform="aou", aou_db_version=8)

cohort.add_covariates(
    sex_at_birth=True,
    drop_nulls=True,
    cohort_file_path="/home/jupyter/df_activity_phe_three.tsv",
    output_file_path="/home/jupyter/cohort_with_covariates_three.tsv"
)
```


```python
df_three = pl.read_csv("/home/jupyter/cohort_with_covariates_three.tsv", separator='\t')
df_three = df_three.with_columns([
    pl.col("date_t0").str.strptime(pl.Date, "%Y-%m-%d").alias("date_t0"),
    pl.col("date_last_ehr").str.strptime(pl.Date, "%Y-%m-%d").alias("date_last_ehr")
])

df_three = df_three.with_columns(
    (
        (pl.col("date_last_ehr") - pl.col("date_t0"))
        .dt.total_days() / 365.25
    ).alias("control_follow_up")
)
```


```python
df_three.write_csv("/home/jupyter/cohort_with_covariates_three.tsv", separator='\t')
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
df = pd.read_csv('/home/jupyter/workspace/Data from All of Us Controlled Tier /workspace-bucket/dataframes/df_per_pid_phetk.csv', parse_dates=['date_t0', 'date_ty', 'date_t0_washout', 'date_last_ehr'])
phecodes = pd.read_csv("/home/jupyter/aou_phecode_counts.tsv", sep="\t")

df_date = df[["person_id", "date_t0", "date_t0_washout"]]

phecodes_date = phecodes.merge(df_date, on="person_id", how="inner")

phecodes_date["first_event_date"] = pd.to_datetime(
    phecodes_date["first_event_date"], format="%Y-%m-%d"
)

phecodes_filtered = phecodes_date[
    phecodes_date["date_t0"] > phecodes_date["first_event_date"]
]

phecodes_filtered.to_csv(
    "aou_phecode_counts_prevalent.tsv",
    sep="\t",
    index=False
)

phecodes_filtered_washout = phecodes_date[
    phecodes_date["date_t0_washout"] > phecodes_date["first_event_date"]
]

phecodes_filtered_washout.to_csv(
    "aou_phecode_counts_prevalent_washout.tsv",
    sep="\t",
    index=False
)
```

Add phecode time to event:


```python
phecode.add_phecode_time_to_event(
    phecode_count_file_path="/home/jupyter/aou_phecode_counts.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    study_start_date_col="date_t0",
    time_unit="years",
    output_file_path="/home/jupyter/phecode_counts_with_time_bmi.tsv"
)

phecode.add_phecode_time_to_event(
    phecode_count_file_path="/home/jupyter/aou_phecode_counts.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    study_start_date_col="date_t0_washout",
    time_unit="years",
    output_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv"
)

phecode.add_phecode_time_to_event(
    phecode_count_file_path="/home/jupyter/aou_phecode_counts.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    study_start_date_col="date_t0",
    time_unit="years",
    output_file_path="/home/jupyter/phecode_counts_with_time_one.tsv"
)

phecode.add_phecode_time_to_event(
    phecode_count_file_path="/home/jupyter/aou_phecode_counts.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    study_start_date_col="date_t0",
    time_unit="years",
    output_file_path="/home/jupyter/phecode_counts_with_time_three.tsv"
)
```

# Run PheWAS

## BMI

### Logistic regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_steps_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence1_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence30_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_dhrps_bmi.tsv"
)

# run PheWAS
phewas.run()
```

### Cox regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_bmi.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_steps_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_bmi.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence1_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_bmi.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence30_bmi.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_bmi.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_bmi.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status", "bmi_cat"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_dhrps_bmi.tsv"
)

# run PheWAS
phewas.run()
```

## Washout

### Logistic regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_steps_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence1_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence30_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_dhrps_washout.tsv"
)

# run PheWAS
phewas.run()
```

### Cox regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0_washout",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_steps_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0_washout",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence1_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0_washout",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence30_washout.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_washout.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_washout.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0_washout",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_washout_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_dhrps_washout.tsv"
)

# run PheWAS
phewas.run()
```

## One year Fitbit

### Logistic regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_steps_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence1_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence30_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_dhrps_one.tsv"
)

# run PheWAS
phewas.run()
```

### Cox regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_one.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_steps_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_one.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence1_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_one.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence30_one.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_one.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_one.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_t0",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_dhrps_one.tsv"
)

# run PheWAS
phewas.run()
```

## Exclude first three years

### Logistic regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_steps_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence1_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_cadence30_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/aou_phecode_counts_prevalent.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="/home/jupyter/phewas_logistic_year_dhrps_three.tsv"
)

# run PheWAS
phewas.run()
```

### Cox regression


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_three.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_ty3",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_steps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_steps_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_three.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_ty3",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence1_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence1_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_three.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_ty3",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_cadence30_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_cadence30_three.tsv"
)

# run PheWAS
phewas.run()
```


```python
%%time
# instantiate class PheWAS object and provide information for the PheWAS run
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="/home/jupyter/phecode_counts_with_time_three.tsv",
    cohort_file_path="/home/jupyter/cohort_with_covariates_three.tsv",
    sex_at_birth_col="sex_at_birth",
    male_as_one=True,
    cox_start_date_col="date_ty3",
    cox_control_observed_time_col="control_follow_up",
    cox_phecode_observed_time_col="phecode_time_to_event",
    covariate_cols=[
        "sex_at_birth", "race_eth", "alcohol", "smoking", "income", "age_t0", "wear_status"
    ],
    independent_variable_of_interest="year_dhrps_scaled",
    min_cases=50,
    min_phecode_count=2,
    method="cox",
    batch_size=8,
    output_file_path="/home/jupyter/phewas_cox_year_dhrps_three.tsv"
)

# run PheWAS
phewas.run()
```
