# Movement test data

Real GPS telemetry used by the animal-movement filter-MLE experiment.

## Dataset

**Hourly GPS location data from mallards, green-winged teal, and American wigeon
transmittered in Arkansas.**
Osborne, D., Askren, R., Phelps, S., et al. (2024). Dryad. Licence: CC0.
https://doi.org/10.5061/dryad.1ns1rn93p

## Download

The loader (`movement_data.py`) expects `hourly_movement.csv` in this directory.
To (re)download the openly-licensed dataset from Dryad:

```bash
cd experiments/movement_test/data
curl -sSL "https://datadryad.org/api/v2/versions/320217/download" -o dryad_bundle.zip
unzip -o dryad_bundle.zip
mv "Hourly_Movement_DabblingDuck_7Oct24.csv" hourly_movement.csv
rm dryad_bundle.zip
```

The raw CSV provides projected UTM coordinates (`Final_X`, `Final_Y`, metres,
UTM 15N) and an hourly-rounded timestamp (`round_cst`); the loader converts to
mean-centred kilometres with time in hours.

These CSVs are not committed to the repo (large); download them on first use.
