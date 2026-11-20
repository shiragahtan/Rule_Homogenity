# Datasets

These three files are the only datasets of the project. They are already filtered, so they can be used
as they are (also outside this code):

| File | Rows | Outcome |
|---|---|---|
| `datasets/german.csv` | 1,000 | `credit_risk` (1 = good) |
| `datasets/stackoverflow.csv` | 37,998 | `ConvertedSalary` |
| `datasets/acs.csv` | 1,188,308 | `Wages or salary income past 12 months` |

They are made by `python scripts/prepare_datasets.py` (the only place where data is filtered):

1. **No missing values.** Every row with an empty cell or `UNKNOWN` is removed
   (German 0, Stack Overflow 92, ACS 0).
2. **10 equal-width bins for every continuous attribute** (the range min..max is cut into 10 intervals of the
   same length; bins can hold very different numbers of rows). Outcomes are not binned; categorical
   attributes are kept as they are.
   - German `age` (years) and `duration` (months): binned from the real numbers of the original UCI
     German Credit data. Labels are the value range:
     age 19-24, 25-30, 31-35, 36-41, 42-47, 48-52, 53-58, 59-63, 64-69, 70-75;
     duration 4-10, 11-17, 18-24, 25-31, 32-38, 39-44, 45-51, 52-58, 59-65, 66-72.
   - German `amount` stays binary, as the instructor asked: 0 = below 4,000 DM (754 rows),
     1 = 4,000 DM or more (246 rows), the same split as the earlier file.
   - ACS `Age`: bin numbers 1-10 in age order:
     1 = 15-23, 2 = 24-31, 3 = 32-39, 4 = 40-47, 5 = 48-55, 6 = 56-63, 7 = 64-71, 8 = 72-79, 9 = 80-87, 10 = 88-95.
   - Stack Overflow: every attribute is a survey category or a survey range (e.g. `25 - 34 years old`);
     there are no raw numbers, so nothing is re-binned.

The earlier unfiltered files (`german_not_encoded.csv`, `stackoverflow_not_encoded.csv`, `acs_encoded.csv`)
are in git history (commit `2c18c55`); the script reads them from there into the git-ignored `datasets/raw/`.

The dataset used by the experiments is chosen in `configs/config.json` (`CHOSEN_DATASET`: so / german / acs).
Rule files (`configs/*Chosen10Treatments.json`) are in `configs/`.
