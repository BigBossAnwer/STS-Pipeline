import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from wn.path import WordNetPaths

from sts_wrldom.corpusReader import read_data
from sts_wrldom.utils import accuracy, get_scores, log_frame, rmse


# Returns a list of predicted labels (floats) utilizing the WordNet semantic tree
#  as laid out in Pawar and Mago, 2018 (http://arxiv.org/abs/1802.05667)
def pawarFit_Predict(docs_disam):
    predictions = []
    for s1_disam, s2_disam in docs_disam:
        s1_tups = [(word, syn) for (word, syn) in s1_disam if syn]
        s2_tups = [(word, syn) for (word, syn) in s2_disam if syn]

        predict = sentence_similarity(s1_tups, s2_tups)
        predictions.append(predict)

    return predictions


# Returns a two tuple, (s1_v, s2_v) which are vector representations of the
#  input sentences using the approach laid out in Pawar and Mago, 2018
def word_similarity(s1, s2, alpha=0.2, beta=0.45):
    wna = WordNetPaths()
    s1_dict = OrderedDict()
    s2_dict = OrderedDict()
    all_syns = s1 + s2
    for (word, syn) in all_syns:
        s1_dict[word] = [0.0]
        s2_dict[word] = [0.0]

    for s1_tup in s1:
        for s2_tup in s2:
            s1_word, s1_syn = s1_tup
            s2_word, s2_syn = s2_tup

            subsumers = wna.lowest_common_hypernyms(
                s1_syn, s2_syn, simulate_root=False, use_min_depth=True
            )
            # In-case subsumers is None
            l = np.inf
            h = 0
            # Now take best subsumer if it exists
            if subsumers:
                subsumer = subsumers[0]
                l = wna.shortest_path_distance(s1_syn, s2_syn, simulate_root=False)
                h = subsumer.max_depth() + 1

            f = np.exp(-alpha * l)
            g1 = np.exp(beta * h)
            g2 = np.exp(-beta * h)
            g = (g1 - g2) / (g1 + g2)
            sim = f * g

            s1_dict[s1_word].append(sim)
            s2_dict[s2_word].append(sim)

    s1_v = np.array([max(s1_dict[word]) for word in s1_dict.keys()])
    s2_v = np.array([max(s2_dict[word]) for word in s2_dict.keys()])

    return s1_v, s2_v


# Returns a predicted label for the sentence pair (as disambiguated (word, synset) lists)
#  passed in. The predicted label is a single float in the range [1, 5]
def sentence_similarity(s1_tups, s2_tups, benchmark_similarity=0.8025, gamma=1.8):
    s1_v, s2_v = word_similarity(s1_tups, s2_tups)

    c1 = 0
    for elem in s1_v:
        if elem > benchmark_similarity:
            c1 += 1
    c2 = 0
    for elem in s2_v:
        if elem > benchmark_similarity:
            c2 += 1

    c_sum = c1 + c2
    if c_sum == 0:
        chi = len(s1_v) / 2
    else:
        chi = c_sum / gamma

    s = np.linalg.norm(s1_v) * np.linalg.norm(s2_v)
    sim = s / chi
    sim = np.clip(sim, 0, 1)
    scaled = (sim * 4) + 1

    return scaled


# Returns a list of two tuples (s1_disam, s2_disam) for each sentence pair in the
# dataframe, where each tuple is a list of disambiguated two tuples (word, synset)
def disambiguate_pipe(df, name=""):
    from pywsd import disambiguate, max_similarity
    from pywsd.lesk import adapted_lesk

    print(f"Disambiguating {name}...")
    disambiguated = []
    for s1, s2, in zip(df["s1"], df["s2"]):
        s1_disam = disambiguate(s1, adapted_lesk, prefersNone=True)
        s2_disam = disambiguate(s2, adapted_lesk, prefersNone=True)
        disambiguated.append((s1_disam, s2_disam))

    return disambiguated


def main():
    description = "WordNet Semantic Tree STS Model (Pawar, 2018)"
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
    parser.add_argument(
        "-p",
        "--pickled",
        help=(
            "Use pre-processed disambiguations (only for given dev and train set), "
            f"pulls from {(Path('pickled/adap_lesk/*'))}"
        ),
        action="store_true",
    )
    args = parser.parse_args()
    log = not args.quiet

    dfs = read_data(["dev", "train"], args.corpus_path, log)
    dev = dfs["dev"]
    train = dfs["train"]

    if args.pickled:
        print("Closed at the moment. Please comeback later!")
        sys.exit()
    else:
        dev_disam = disambiguate_pipe(dev, "Dev")
        train_disam = disambiguate_pipe(train, "Train")

    dev_predics = pawarFit_Predict(dev_disam)
    train_predics = pawarFit_Predict(train_disam)

    dev["pawarPredics"] = [int(elem) for elem in np.round(dev_predics)]
    train["pawarPredics"] = [int(elem) for elem in np.round(train_predics)]

    if log:
        for df, name in zip([dev, train], ["dev", "train"]):
            log_frame(df, name=name, tag="pawar_predics")

    for df, name in zip([dev, train], ["Dev", "Train"]):
        acc = accuracy(df["pawarPredics"], df["gold"])
        _rmse = rmse(df["pawarPredics"], df["gold"])
        pear_corr = pearsonr(list(df["pawarPredics"]), list(df["gold"]))
        cols = ["RMSE", "Accuracy", "Pearson's R", "Pearson's R p-val"]
        vals = [_rmse, acc, pear_corr[0], pear_corr[1]]
        stats = pd.DataFrame(
            list(df["pawarPredics"]), columns=["Predic_Label"]
        ).describe()
        extra = pd.DataFrame(vals, index=cols, columns=["Predic_Label"])
        print(f"\n{name} Gold stats: ")
        print(pd.DataFrame(list(df["gold"]), columns=["Gold_Label"]).describe().T)
        print(f"\n{name} Pawar Model Prediction stats: ")
        print(stats.append(extra).T)
        print("\n------")

    for df, name in zip([dev, train], ["Dev", "Train"]):
        print(f"\n{name} Prediction Metrics:")
        metrics = get_scores(list(df["pawarPredics"]), list(df["gold"]))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
