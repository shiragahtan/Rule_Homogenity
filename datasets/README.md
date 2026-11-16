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
2. **10 equal-frequency bins for every numeric attribute** (outcomes are not binned; categorical
   attributes are kept as they are):
   - German `duration` (months), `amount` (DM), `age` (years): binned from the real numbers of the
     original UCI German Credit data. Labels are the value range, e.g. `age = 19-23`.
     `duration` has only 8 bins because many loans share the same length (12, 24 months, ...).
   - ACS `Age`: bin numbers 1-10 in age order:
     1 = 15-21, 2 = 22-28, 3 = 29-35, 4 = 36-42, 5 = 43-49, 6 = 50-55, 7 = 56-61, 8 = 62-67, 9 = 68-74, 10 = 75-95.
   - Stack Overflow: every attribute is a survey category or a survey range (e.g. `25 - 34 years old`);
     there are no raw numbers, so nothing is re-binned.

The earlier unfiltered files (`german_not_encoded.csv`, `stackoverflow_not_encoded.csv`, `acs_encoded.csv`)
are in git history (commit `2c18c55`); the script reads them from there into the git-ignored `datasets/raw/`.

The dataset used by the experiments is chosen in `configs/config.json` (`CHOSEN_DATASET`: so / german / acs).
Rule files (`configs/*Chosen10Treatments.json`) are in `configs/`.
