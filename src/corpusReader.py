import argparse
import sys
from pathlib import Path

import pandas as pd


def read_data(which_sets, corpus_path=None, log=False):
    data_files = {"dev": "dev-set.txt", "train": "train-set.txt", "test": "test-set.txt"}
    data_dir = corpus_path if corpus_path is not None else "data"
    data_frames = {}

    if "dev" in which_sets:
        try:
            dev_path = Path(data_dir, data_files["dev"])
            print(f"\nReading dev-set from: {dev_path}")
            dev = []
            rows = 0
            with open(dev_path) as dev_fh:
                next(dev_fh) # Skip header
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
            dev_set = dev_set.dropna()
            pairs_omitted = rows - dev_set.shape[0]
            print(f"Dev DF shape: {dev_set.shape}")
            print(f"Dev Pairs Omitted: {pairs_omitted} = {rows} - {dev_set.shape[0]}")
            data_frames["dev"] = dev_set

        except FileNotFoundError:
            print(f"Error: No such data file at found at {dev_path}")
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    if "train" in which_sets:
        try:
            train_path = Path(data_dir, data_files["train"])
            print(f"\nReading train-set from: {train_path}")
            train = []
            rows = 0
            with open(train_path) as train_fh:
                next(train_fh) # Skip header
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
            train_set = train_set.dropna()
            pairs_omitted = rows - train_set.shape[0]
            print(f"Train DF shape: {train_set.shape}")
            print(f"Train Pairs Omitted: {pairs_omitted} = {rows} - {train_set.shape[0]}")
            data_frames["train"] = train_set

        except FileNotFoundError:
            print(f"Error: No such data file at found at {train_path}")
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    if "test" in which_sets:
        try:
            test_path = Path(data_dir, data_files["test"])
            print(f"\nReading test-set from: {test_path}")
            test = []
            rows = 0
            with open(test_path) as test_fh:
                next(test_fh) # Skip header
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
            test_set = test_set.dropna()
            pairs_omitted = rows - test_set.shape[0]
            print(f"Test DF shape: {test_set.shape}")
            print(f"Test Pairs Omitted: {pairs_omitted} = {rows} - {test_set.shape[0]}")
            data_frames["test"] = test_set

        except FileNotFoundError:
            print(f"Error: No such data file at found at {test_path}")
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise
    
    if log:
        try:
            Path("log").mkdir(exist_ok=True)
            for frame in data_frames.keys():
                data_frames[frame].to_csv(str(Path("log", frame + ".csv")))

        except IOError:
            print("Error: Log write failed")
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise
    
    return data_frames

def main():
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
    parser.add_argument(
        "-l",
        "--log",
        help=(
            "Suppress dataframe logging to "
            + str(Path("out"))
        ),
        action="store_true"
    )
    args = parser.parse_args()
    
    read_data(["dev", "test", "train"], args.corpus_path, args.log)

if __name__== "__main__":
  main()
