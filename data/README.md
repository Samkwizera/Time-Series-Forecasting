# Data directory

Nothing here is tracked by git except this file.

```
data/
  raw/         62 daily TSV files from Harvard Dataverse (20.8 GB) - download manually, see ../README.md
  interim/     one Parquet file per day, country codes summed  (scripts/01_ingest.py)
  processed/   hourly (time x 10,000 cells) matrices per activity + 10-min citywide totals
  smoke/       synthetic fixture created by `--synthetic-days`; safe to delete
```
