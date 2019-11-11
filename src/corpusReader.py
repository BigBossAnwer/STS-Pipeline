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
    dev_set = pd.read_table(
        dev_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2", "gold"],
        dtype={
            "id": object,
            "Sentence1": object,
            "Sentence2": object,
            "Gold Tag": pd.Int64Dtype(),
        },
    )
    with open(dev_path) as dev_fh:
        raw_pairs = len(dev_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {dev_path}")

drop_dev_set = dev_set.dropna()
pairs_omitted = raw_pairs - drop_dev_set.shape[0]
print(dev_set.head())
print(f"Raw dev DF shape: {dev_set.shape}")
print(f"NA dropped dev DF shape: {drop_dev_set.shape}")
print(f"Dev Pairs Omitted: {pairs_omitted} = {raw_pairs} - {drop_dev_set.shape[0]}")

try:
    train_path = Path(data_dir, data_files["train"])
    print(f"\nReading train-set from: {train_path}")
    train_set = pd.read_table(
        train_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2", "gold"],
        dtype={
            "id": object,
            "Sentence1": object,
            "Sentence2": object,
            "Gold Tag": pd.Int64Dtype(),
        },
    )
    with open(train_path) as train_fh:
        raw_pairs = len(train_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {train_path}")

drop_train_set = train_set.dropna()
pairs_omitted = raw_pairs - drop_train_set.shape[0]
print(train_set.head())
print(f"Raw train DF shape: {train_set.shape}")
print(f"NA dropped train DF shape: {drop_train_set.shape}")
print(f"Train Pairs Omitted: {pairs_omitted} = {raw_pairs} - {drop_train_set.shape[0]}")

try:
    test_path = Path(data_dir, data_files["test"])
    print(f"\nReading test-set from: {test_path}")
    test_set = pd.read_table(
        test_path,
        error_bad_lines=False,
        header=0,
        names=["id", "s1", "s2"],
        dtype={"id": object, "Sentence1": object, "Sentence2": object},
    )
    with open(test_path) as test_fh:
        raw_pairs = len(test_fh.readlines()) - 1

except FileNotFoundError:
    print(f"Error: No such data file at found at {test_path}")

drop_test_set = test_set.dropna()
pairs_omitted = raw_pairs - drop_test_set.shape[0]
print(test_set.head())
print(f"Raw test DF shape: {train_set.shape}")
print(f"NA dropped test DF shape: {drop_test_set.shape}")
print(f"Test Pairs Omitted: {pairs_omitted} = {raw_pairs} - {drop_test_set.shape[0]}")

Path("out").mkdir(exist_ok=True)
try:
    drop_dev_set.to_csv(str(Path("out/dev_df.txt")), sep="\t")
    drop_train_set.to_csv(str(Path("out/train_df.txt")), sep="\t")
    drop_test_set.to_csv(str(Path("out/test_df.txt")), sep="\t")

except IOError:
    print("Error: Log write failed")
    sys.exit()
