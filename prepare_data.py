"""Deterministic generator of data versions for the data-versioning practice.

Modes:
  base           -> data/iris.csv  (original Iris)
  wine           -> data/wine.csv  (original Wine, Iris is not touched)
  iris_features  -> data/iris.csv  (Iris + sepal_area, petal_area, Wine is not touched)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = Path("data")
AREA_DECIMALS = 4


def load_frame(loader) -> pd.DataFrame:
    """Load a scikit-learn dataset as a frame sorted by ALL columns (features + target)."""
    frame = loader(as_frame=True).frame
    return frame.sort_values(by=list(frame.columns), kind="stable").reset_index(drop=True)


def add_iris_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sepal_area"] = (frame["sepal length (cm)"] * frame["sepal width (cm)"]).round(AREA_DECIMALS)
    frame["petal_area"] = (frame["petal length (cm)"] * frame["petal width (cm)"]).round(AREA_DECIMALS)
    return frame


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    # identical saving parameters for every version
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=["base", "wine", "iris_features"])
    args = parser.parse_args()

    if args.mode == "base":
        path = save_csv(load_frame(load_iris), "iris.csv")
    elif args.mode == "wine":
        path = save_csv(load_frame(load_wine), "wine.csv")
    else:
        path = save_csv(add_iris_features(load_frame(load_iris)), "iris.csv")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
