# Data

`AXD_measured.csv` is the subset of the AXD localisation dataset used in the
paper: every trial of the **Measured** condition, in which listeners
localised sounds rendered with their own measured HRTF. It was extracted from
`AXD_full_dataset.csv` with [`make_subset.py`](make_subset.py), which drops
the HRTF file name, headphone model, headphone equalisation and time-stamp
columns.

## Contents

| | |
|---|---|
| Trials | 5742 |
| Listeners | 34 unique participant ids (18 from Daugintis2025, 20 from Pirard2025, 20 from Poole2025) |
| Trials per listener | 99 to 297 |

Some listeners took part in more than one of the three experiments. The
notebooks pool a listener's trials across experiments by participant id, as
in the paper. Participant `P0247` is excluded by the notebooks, leaving the
33 listeners reported in the paper.

<!-- TODO: state the reason P0247 is excluded. -->

## Columns

All angles are in degrees.

| Column | Description |
|---|---|
| `experiment` | Source experiment: `Daugintis2025`, `Pirard2025` or `Poole2025` |
| `condition` | Always `Measured` in this file |
| `participant` | Listener id |
| `hrtf_id` | Id of the HRTF used for rendering (the listener's own) |
| `repetition`, `trial` | Block and trial counters |
| `azi_target`, `ele_target` | Target direction, spherical coordinates (azimuth, elevation) |
| `azi_response`, `ele_response` | Response direction, spherical coordinates |
| `lat_target`, `pol_target` | Target direction, horizontal-polar coordinates (lateral in [-90, 90], polar in [-90, 270)) |
| `lat_response`, `pol_response` | Response direction, horizontal-polar coordinates |
| `great_circle_error` | Angle between target and response directions |

The mixture model uses `lat_target`, `pol_target` and `pol_response`; the
classical metrics use the four horizontal-polar columns.

## Provenance and citation

The AXD dataset was collected at the Audio Experience Design group, Imperial
College London, within the SONICOM project. If you use these data, please
cite the original publications.

<!-- TODO: add the references for the three source experiments
     (Daugintis et al. 2025, Pirard et al. 2025, Poole et al. 2025) and the
     dataset DOI / licence. -->
