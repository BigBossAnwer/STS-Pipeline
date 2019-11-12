import argparse
import sys
from pathlib import Path

import pandas as pd

description = (
    f"Eats project data as specified by the structure found in " f"{Path('data/*.csv')}"
)
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-c",
    "--corpus_path",
    help=(str(Path("Path/To/Corpus/*.csv")) + ", Default: " + str(Path("data/*.csv"))),
)
args = parser.parse_args()

data_files = {"dev": "dev.csv", "train": "train.csv", "test": "test.csv"}
data_dir = args.corpus_path if args.corpus_path is not None else "data"

try:
    dev_path = Path(data_dir, data_files["dev"])
    orig_dev_path = Path(data_dir, "dev-set.txt")
    print(f"Reading dev-set from: {dev_path}")
    dev_set = pd.read_csv(
        dev_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2", "gold"],
        dtype={"id": object, "Sentence1": object, "Sentence2": object, "Gold Tag": int,},
        skip_blank_lines=True,
    )
    with open(orig_dev_path) as dev_fh:
        raw_pairs = len(dev_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {dev_path}")

dev_set = dev_set.dropna()
pairs_omitted = raw_pairs - dev_set.shape[0]
print(dev_set.head())
print(f"Dev DF shape: {dev_set.shape}")
print(f"Dev Pairs Omitted: {pairs_omitted} = {raw_pairs} - {dev_set.shape[0]}")

try:
    train_path = Path(data_dir, data_files["train"])
    orig_train_path = Path(data_dir, "train-set.txt")
    print(f"\nReading train-set from: {train_path}")
    train_set = pd.read_csv(
        train_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2", "gold"],
        dtype={"id": object, "Sentence1": object, "Sentence2": object, "Gold Tag": int,},
        skip_blank_lines=True,
    )
    with open(orig_train_path) as train_fh:
        raw_pairs = len(train_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {train_path}")

train_set = train_set.dropna()
pairs_omitted = raw_pairs - train_set.shape[0]
print(train_set.head())
print(f"Train DF shape: {train_set.shape}")
print(f"Train Pairs Omitted: {pairs_omitted} = {raw_pairs} - {train_set.shape[0]}")

try:
    test_path = Path(data_dir, data_files["test"])
    orig_test_path = Path(data_dir, "test-set.txt")
    print(f"\nReading test-set from: {test_path}")
    test_set = pd.read_csv(
        test_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2"],
        dtype={"id": object, "Sentence1": object, "Sentence2": object},
        skip_blank_lines=True,
    )
    with open(orig_test_path) as test_fh:
        raw_pairs = len(test_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {test_path}")

test_set = test_set.dropna()
pairs_omitted = raw_pairs - test_set.shape[0]
print(test_set.head())
print(f"Test DF shape: {train_set.shape}")
print(f"Test Pairs Omitted: {pairs_omitted} = {raw_pairs} - {test_set.shape[0]}")

Path("out").mkdir(exist_ok=True)
try:
    dev_set.to_csv(str(Path("out/dev_df.csv")))
    train_set.to_csv(str(Path("out/train_df.csv")))
    test_set.to_csv(str(Path("out/test_df.csv")))

except IOError:
    print("Error: Log write failed")
    sys.exit()
