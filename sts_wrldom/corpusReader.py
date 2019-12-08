import argparse
import sys
from pathlib import Path

import pandas as pd

from sts_wrldom.utils import log_frame


def read_data(which_sets, corpus_path="data", log=False):
    """Returns the specified data sets as dataframes

    Args:
        which_sets (list): the specified data sets.
            ex: ["dev", "train"] or ["train"]
        corpus_path (str, optional): path to the parent dir housing the data.
        log (bool, optional): specifies logging of the produced dataframe as 
            a .csv to "./log"

    Returns:
        if only one data set was specified:
            the data set specified as a dataframe
        else:
            a dictionary with all the requested data sets as dataframes like:
            {
                "dev": dev_df,
                "train": train_df,
                ...
            }
    """
    if not corpus_path:
        corpus_path = "data"
    data_files = {"dev": "dev-set.txt", "train": "train-set.txt", "test": "test-set.txt"}
    data_frames = {}

    if "dev" in which_sets:
        try:
            dev_path = Path(corpus_path, data_files["dev"])
            print(f"Reading dev-set from: {dev_path}")
            dev = []
            rows = 0
            with open(dev_path, encoding="utf8") as dev_fh:
                next(dev_fh)  # Skip header
                for line in dev_fh.readlines():
                    rows += 1
                    cleaned = []
                    for item in line.split("\t"):
                        # Clean hanging new line trailing Gold Tag
                        clean = item.split("\n")
                        cleaned.append(clean[0])
                    dev.append(cleaned)

            dev_set = pd.DataFrame(dev, columns=["id", "s1", "s2", "gold"]).astype(
                dtype={"id": object, "s1": object, "s2": object, "gold": int,}
            )
            dev_set = dev_set.dropna()
            pairs_omitted = rows - dev_set.shape[0]
            print(f"Dev DF shape: {dev_set.shape}")
            print(f"Dev Pairs Omitted: {pairs_omitted} = {rows} - {dev_set.shape[0]}\n")
            data_frames["dev"] = dev_set

        except FileNotFoundError:
            print(
                f"Error: No such data file at found at {str(Path(Path.cwd(), dev_path))}\n"
            )
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    if "train" in which_sets:
        try:
            train_path = Path(corpus_path, data_files["train"])
            print(f"Reading train-set from: {train_path}")
            train = []
            rows = 0
            with open(train_path, encoding="utf8") as train_fh:
                next(train_fh)  # Skip header
                for line in train_fh.readlines():
                    rows += 1
                    cleaned = []
                    for item in line.split("\t"):
                        # Clean hanging new line trailing Gold Tag
                        clean = item.split("\n")
                        if clean[0]:
                            cleaned.append(clean[0])
                    train.append(cleaned)

            train_set = pd.DataFrame(train, columns=["id", "s1", "s2", "gold"]).astype(
                dtype={"id": object, "s1": object, "s2": object, "gold": int,}
            )
            train_set = train_set.dropna()
            pairs_omitted = rows - train_set.shape[0]
            print(f"Train DF shape: {train_set.shape}")
            print(
                f"Train Pairs Omitted: {pairs_omitted} = {rows} - {train_set.shape[0]}\n"
            )
            data_frames["train"] = train_set

        except FileNotFoundError:
            print(
                f"Error: No such data file at found at {str(Path(Path.cwd(), train_path))}\n"
            )
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    if "test" in which_sets:
        try:
            test_path = Path(corpus_path, data_files["test"])
            print(f"Reading test-set from: {test_path}")
            test = []
            rows = 0
            with open(test_path, encoding="utf8") as test_fh:
                next(test_fh)  # Skip header
                for line in test_fh.readlines():
                    rows += 1
                    cleaned = []
                    for item in line.split("\t"):
                        # Clean hanging new line trailing Gold Tag
                        clean = item.split("\n")
                        cleaned.append(clean[0])
                    test.append(cleaned)

            test_set = pd.DataFrame(test, columns=["id", "s1", "s2"]).astype(
                dtype={"id": object, "s1": object, "s2": object,}
            )
            test_set = test_set.dropna()
            pairs_omitted = rows - test_set.shape[0]
            print(f"Test DF shape: {test_set.shape}")
            print(f"Test Pairs Omitted: {pairs_omitted} = {rows} - {test_set.shape[0]}\n")
            data_frames["test"] = test_set

        except FileNotFoundError:
            print(
                f"Error: No such data file at found at {str(Path(Path.cwd(), test_path))}\n"
            )
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

    if len(which_sets) == 1:
        return next(iter(data_frames.values()))  # returns first (and only) dataframe
    else:
        return data_frames


def main():
    description = (
        f"Eats project data as specified by the structure found in "
        f"{Path('data/*-set.txt')}. Running this standalone amounts to testing"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--corpus_path",
        help=f"{Path('Path/To/Corpus/*-set.txt')}, Default: {Path('data/*-set.txt')}",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help=f"Suppresses logging of produced log files to: {Path('log/*')}",
        action="store_true",
    )
    args = parser.parse_args()
    log = not args.quiet

    dfs = read_data(["dev", "test", "train"], args.corpus_path, log)

    for frame in dfs.keys():
        frame_cap = frame[0].upper() + frame[1:]
        print(frame_cap + " head: ")
        print(dfs[frame].head(), "\n")
        if frame in ["dev", "train"]:
            print(frame_cap + " gold tag stats: ")
            print(dfs[frame]["gold"].describe().to_frame().T, "\n")
            print(frame_cap + " gold tag counts: ")
            print(dfs[frame]["gold"].value_counts().to_frame().T, "\n")


if __name__ == "__main__":
    main()
