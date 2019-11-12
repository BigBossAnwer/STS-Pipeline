import argparse
import sys
from pathlib import Path

import pandas as pd

description = (
    f"Eats project data as specified by the structure found in "
    f"{Path('data/*-set.txt')}"
)
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-c",
    "--corpus_path",
    help=(
        str(Path("Path/To/Corpus/*-set.txt"))
        + ", Default: "
        + str(Path("data/*-set.txt"))
    ),
)
args = parser.parse_args()

data_files = {"dev": "dev-set.txt", "train": "train-set.txt", "test": "test-set.txt"}
data_dir = args.corpus_path if args.corpus_path is not None else "data"

try:
    dev_path = Path(data_dir, data_files["dev"])
    print(f"Reading dev-set from: {dev_path}")
    dev = []
    rows = 0
    with open(dev_path) as dev_fh:
        next(dev_fh)
        for line in dev_fh.readlines():
            rows += 1
            cleaned = []
            for item in line.split("\t"):
                clean = item.split("\n") # Clean hanging new line trailing Gold Tag
                cleaned.append(clean[0])
            dev.append(cleaned)

    dev_set = pd.DataFrame(dev, columns=["id", "s1", "s2", "gold"]).astype(
        dtype={"id": object, "s1": object, "s2": object, "gold": int,}
    )

except FileNotFoundError:
    print(f"Error: No such data file at found at {dev_path}")

dev_set = dev_set.dropna()
pairs_omitted = rows - dev_set.shape[0]
print(dev_set.head())
print(f"Dev DF shape: {dev_set.shape}")
print(f"Dev Pairs Omitted: {pairs_omitted} = {rows} - {dev_set.shape[0]}")

try:
    train_path = Path(data_dir, data_files["train"])
    print(f"Reading train-set from: {train_path}")
    train = []
    rows = 0
    with open(train_path) as train_fh:
        next(train_fh)
        for line in train_fh.readlines():
            rows += 1
            cleaned = []
            for item in line.split("\t"):
                clean = item.split("\n") # Clean hanging new line trailing Gold Tag
                if clean[0]:
                    cleaned.append(clean[0])
            train.append(cleaned)
    
    train_set = pd.DataFrame(train, columns=["id", "s1", "s2", "gold"]).astype(
        dtype={"id": object, "s1": object, "s2": object, "gold": int,}
    )

except FileNotFoundError:
    print(f"Error: No such data file at found at {train_path}")

train_set = train_set.dropna()
pairs_omitted = rows - train_set.shape[0]
print(train_set.head())
print(f"Train DF shape: {train_set.shape}")
print(f"Train Pairs Omitted: {pairs_omitted} = {rows} - {train_set.shape[0]}")

try:
    test_path = Path(data_dir, data_files["test"])
    print(f"\nReading test-set from: {test_path}")
    test = []
    rows = 0
    with open(test_path) as test_fh:
        next(test_fh)
        for line in test_fh.readlines():
            rows += 1
            cleaned = []
            for item in line.split("\t"):
                clean = item.split("\n") # Clean hanging new line trailing Gold Tag
                cleaned.append(clean[0])
            test.append(cleaned)

    test_set = pd.DataFrame(test, columns=["id", "s1", "s2"]).astype(
        dtype={"id": object, "s1": object, "s2": object,}
    )

except FileNotFoundError:
    print(f"Error: No such data file at found at {test_path}")

test_set = test_set.dropna()
pairs_omitted = rows - test_set.shape[0]
print(test_set.head())
print(f"Test DF shape: {test_set.shape}")
print(f"Test Pairs Omitted: {pairs_omitted} = {rows} - {test_set.shape[0]}")

Path("out").mkdir(exist_ok=True)
try:
    dev_set.to_csv(str(Path("out/dev.csv")))
    train_set.to_csv(str(Path("out/train.csv")))
    test_set.to_csv(str(Path("out/test.csv")))

except IOError:
    print("Error: Log write failed")
    sys.exit()
