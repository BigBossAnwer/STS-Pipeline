import argparse
from pathlib import Path

import pandas
import spacy

from corpusReader import log_frame, read_data


def preprocess(df, name=None, tag=None, log=False):
    nlp = spacy.load("en_core_web_sm")

    if name is None:
        announce = "Enriching dataframe with tokens, lemmas, and POS tags..."
    else:
        announce = f"Enriching {name} dataframe with tokens, lemmas, and POS tags..."
    print(announce)
    parse_cols = ["s1", "s2"]
    for col in parse_cols:
        tokens = []
        lemma = []
        pos = []
        parse_fail = 0

        for doc in nlp.pipe(df[col].values, batch_size=50, n_threads=4):
            if doc.is_parsed:
                tokens.append([n.text for n in doc])
                lemma.append([n.lemma_ for n in doc])
                pos.append([n.pos_ for n in doc])
            else:
                # Ensure parse lists have the same number of entries as the original
                #   Dataframe regardless of parse failure
                parse_fail += 1
                tokens.append(None)
                lemma.append(None)
                pos.append(None)

        print(f"{col.upper()} parse failures: {parse_fail}")
        df[col + "_tokens"] = tokens
        df[col + "_lemma"] = lemma
        df[col + "_pos"] = pos

    if log:
        if name is None or tag is None:
            print("Error: Logging requires a name and a tag")
        else:
            log_frame(df, name=name, tag=tag)
    print()

    return df


def main():
    description = (
        f"Pipeline to enrich project data as specified by the dataframe structure found "
        f"in {Path('src/corpusReader.py')}"
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
        help=("Log produced Dataframes to " + str(Path("log/*"))),
        action="store_true",
    )
    args = parser.parse_args()

    if args.corpus_path is None:
        dfs = read_data(["dev", "train", "test"], log=args.log)
    else:
        dfs = read_data(["dev", "train", "test"], args.corpus_path, args.log)

    for frame in dfs.keys():
        preprocess(dfs[frame], name=frame, tag="enriched", log=args.log)

    for frame in dfs.keys():
        print("Enriched " + frame + " head: ")
        print(dfs[frame].head(), "\n")


if __name__ == "__main__":
    main()
