"""
Extract the subset of the AXD dataset that is shipped with this repository.

The notebooks use only the *Measured* condition (listeners localising with
their own measured HRTF). This script keeps those rows, drops the columns
that are irrelevant for the analyses (HRTF file name, headphone model and
equalisation, time stamp) and writes ``data/AXD_measured.csv`` next to
this file.

Usage
-----
    python data/make_subset.py /path/to/AXD_full_dataset.csv
"""
import pathlib
import sys

import pandas as pd

DROP_COLUMNS = ['hrtf_file', 'headphones', 'headphones_eq', 'time']


def main(src):
    df = pd.read_csv(src)
    df = df[df['condition'] == 'Measured']
    df = df.drop(columns=DROP_COLUMNS).reset_index(drop=True)
    out = pathlib.Path(__file__).with_name('AXD_measured.csv')
    df.to_csv(out, index=False)
    print(f"{len(df)} trials, {df['participant'].nunique()} participants "
          f"-> {out}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
