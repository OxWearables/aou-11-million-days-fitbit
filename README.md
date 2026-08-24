# 11 million days of longitudinal Fitbit data reveal novel future health insights


## Overview

This repository contains the analysis pipeline used to characterize associations between wearable-derived physical activity metrics and disease phenotypes (phecodes) in the *All of Us* Research Program. The pipeline covers data import and cleaning, phenome-wide association study (PheWAS) analyses (primary and sensitivity), and generation of all manuscript tables and figures.

The repository consists of four scripts, intended to be run in sequence.

---

## Repository structure

| File | Language | Purpose |
|---|---|---|
| `1_data-import.Rmd` | R | Imports raw variables from AoU via SQL/BigQuery, cleans and processes covariates and Fitbit-derived exposures, applies exclusion criteria, and derives time-windowed activity phenotypes |
| `2a_PheWAS.md` | Python | Runs the primary PheWAS (logistic regression for prevalent associations; Cox regression for incident associations) across step count, peak 1-min cadence, peak 30-min cadence, and daily heart rate per step metrics at multiple time windows |
| `2b_PheWAS-sensitivity.md` | Python | Runs sensitivity PheWAS analyses (adjusting for BMI, applying a washout period, restricting to one year of Fitbit data, and excluding the first three years of follow-up) |
| `3_tables-figures.Rmd` | R | Loads all PheWAS outputs and generates the manuscript's tables and figures |

---

## Analysis details

### 1. Data import and cleaning (`1_data-import.Rmd`)

- Pulls covariates (sex, race/ethnicity, income, alcohol, smoking, BMI category, age) and EHR observation windows via SQL.
- Imputes missing covariates using `mice` (excluding sex, which is required and not imputed).
- Pulls Fitbit-derived daily metrics: step count, peak 1-minute cadence, peak 30-minute cadence, and daily heart rate per step (DHRPS), plus wear time and WEAR/BYOD status.
- Applies a sequential exclusion pipeline (age, physiologically implausible values, wear time).
- Defines index dates and time windows (day, week, month, six months, year) plus washout variants, and computes person-level summary phenotypes and quartiles.

### 2. Primary PheWAS (`2a_PheWAS.ipynb`)

- Uses the [PheTK](https://github.com/nhgritctran/PheTK) package.
- Runs logistic regression PheWAS (prevalent cases) and Cox regression PheWAS (incident cases) for each of steps, cadence1, cadence30, and DHRPS, across five time windows (day/week/month/six months/year).
- Produces Manhattan plots and per-analysis result files.

### 3. Sensitivity PheWAS (`2b_PheWAS-sensitivity.ipynb`)

- Uses the [PheTK](https://github.com/nhgritctran/PheTK) package.
- Repeats the year-window PheWAS under four sensitivity conditions: additional BMI adjustment, a 14-day washout period, restriction to participants with ≥1 year of Fitbit data, and exclusion of the first three years of follow-up.
- Outputs mirror the primary analysis structure for direct comparison.

### 4. Tables and figures (`3_tables-figures.Rmd`)

- Loads primary and sensitivity PheWAS results and combines them into unified data frames.
- Generates all main and supplementary figures.

---

## Execution order

```
1_data-import.Rmd
      │
      ▼
2a_PheWAS.ipynb
      │
      ▼
2b_PheWAS-sensitivity.ipynb
      │
      ▼
3_tables-figures.Rmd
```

---

## Notes and caveats

- File paths reference a standard AoU Workbench bucket structure; update local paths as needed if adapting outside this environment.
- The kernel must be restarted after installing `PheTK` before importing its submodules (noted inline in the notebooks).
- Bonferroni thresholds are computed per analysis using the number of tested phecodes in that specific run, not a single global constant.
- Because this uses AoU Controlled Tier data, no participant-level data or rendered outputs containing cell counts below the AoU minimum cell size threshold (n < 20) should be exported outside the Workbench or committed to a public repository, consistent with AoU data use policies.
