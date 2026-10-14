"""Validate CSV files in data/ and print a short report.

Exit code != 0 if:
  - an expected file is missing,
  - an unnamed (pandas index) column is found,
  - the number of rows differs from the scikit-learn dataset,
  - original columns differ from scikit-learn or computed columns are wrong.

Usage: python check_data.py [--expect iris wine]   (default: iris)
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = Path("data")
LOADERS = {"iris": load_iris, "wine": load_wine}
IRIS_EXTRA = ["sepal_area", "petal_area"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(name: str) -> pd.DataFrame:
    return LOADERS[name](as_frame=True).frame


def check_table(name: str, frame: pd.DataFrame) -> list[str]:
    errors = []
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if unnamed:
        errors.append(f"{name}: unnamed index-like columns {unnamed}")

    ref = reference(name)
    if len(frame) != len(ref):
        errors.append(f"{name}: expected {len(ref)} rows, got {len(frame)}")

    base_cols = list(ref.columns)
    if list(frame.columns[: len(base_cols)]) != base_cols:
        errors.append(f"{name}: original columns lost or reordered")
        return errors

    if len(frame) == len(ref):
        key = list(ref.columns)
        ref_sorted = ref.sort_values(by=key, kind="stable").reset_index(drop=True)
        if not np.allclose(frame[base_cols].to_numpy(float), ref_sorted.to_numpy(float)):
            errors.append(f"{name}: original values differ from scikit-learn")

    extra = list(frame.columns[len(base_cols):])
    if name == "iris":
        if extra not in ([], IRIS_EXTRA):
            errors.append(f"iris: unexpected extra columns {extra}")
        if extra == IRIS_EXTRA:
            sepal = frame["sepal length (cm)"] * frame["sepal width (cm)"]
            petal = frame["petal length (cm)"] * frame["petal width (cm)"]
            if not np.allclose(frame["sepal_area"], sepal, atol=1e-4):
                errors.append("iris: sepal_area is wrong")
            if not np.allclose(frame["petal_area"], petal, atol=1e-4):
                errors.append("iris: petal_area is wrong")
    elif extra:
        errors.append(f"{name}: unexpected extra columns {extra}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", nargs="+", default=["iris"], choices=list(LOADERS))
    args = parser.parse_args()

    errors = [f"missing expected file {DATA_DIR / (n + '.csv')}" for n in args.expect
              if not (DATA_DIR / f"{n}.csv").exists()]

    for path in sorted(DATA_DIR.glob("*.csv")):
        name = path.stem
        frame = pd.read_csv(path)
        print(f"{path.as_posix()}")
        print(f"  sha256 : {sha256(path)}")
        print(f"  shape  : {frame.shape[0]} rows x {frame.shape[1]} cols")
        print(f"  columns: {list(frame.columns)}")
        print(f"  missing: {int(frame.isna().sum().sum())}")
        if name in LOADERS:
            errors += check_table(name, frame)
        else:
            errors.append(f"unknown file {path.name}")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
