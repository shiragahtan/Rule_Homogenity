---
name: prepare-datasets
description: Rebuild or change the clean dataset CSVs (datasets/german.csv, stackoverflow.csv, acs.csv) of this project - missing values, binning, the binary German amount. Use when the data filtering or binning must change, or someone asks how the datasets were made.
---

# Prepare the datasets

`scripts/prepare_datasets.py` is the ONLY place where data is filtered. The CATE code and the algorithms read the
clean files as they are. Never add filtering to `utils/ATE_update.py`.

```
python scripts/prepare_datasets.py
```

## Rules (decided by the instructor, Brit)
1. Remove every row with an empty cell or `UNKNOWN` (German 0, SO 92, ACS 0 rows removed).
2. 10 **equal-width** bins (`pd.cut`, same range per bin) for continuous attributes only:
   German `age` and `duration` (labels are the ranges, e.g. `19-24`), ACS `Age` (codes 1-10 in age order).
   Not equal-frequency (`pd.qcut`): that was an earlier version and was rejected.
3. German `amount` stays binary: 0 = below 4,000 DM (754 rows), 1 = 4,000 DM or more (246). The instructor
   said binary attributes are not binned.
4. Outcomes and categorical attributes are not changed. SO has only survey ranges, so nothing is rebinned.

## Inputs
- German real numbers: UCI Statlog `german.data` (downloaded automatically). The row order matches the earlier
  file; the script asserts it on 5 shared columns.
- SO and ACS: the earlier files from git history (commit `2c18c55`), copied into the git-ignored `datasets/raw/`.

## After a change
- Update `datasets/README.md` (bin ranges and row counts printed by the script) and the Datasets section of CLAUDE.md.
- Every result depends on the data: rule lists and experiments must be rebuilt (skills `select-rules`,
  `full-experiments`). Say clearly in any summary which results predate the change.
- Commit message: list the dataset paths, so the teammates (e.g. Sean) know which files to use.
