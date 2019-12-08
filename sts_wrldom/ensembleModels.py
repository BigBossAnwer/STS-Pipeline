import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sts_wrldom.corpusReader import read_data
from sts_wrldom.depTFIDFModel import depFit_Predict
from sts_wrldom.enrichPipe import preprocess_raw
from sts_wrldom.pawarModel import disambiguate_pipe, pawarFit_Predict
from sts_wrldom.utils import accuracy, get_scores, log_frame, rmse, write_results


def ensemble_head(dep_preds, pawar_preds, embed_preds=None, **kwargs):
    """Returns a weighted combination of all sts_wrldom sub-models. Weights were defined
    by grid-search testing (see STS-Project/notebooks/ensembleParamTest.ipynb).
    
    Args:
        dep_preds (list): a list of predicted labels (float) in range [1, 5] from the
            Dependency Tree TFIDF model (depTFIDFModel).
        pawar_preds (list): a list of predicted labels (float) in range [1, 5] from the
            WordNet Features / Pawar model (pawarModel).
        embed_preds (list, optional): a list of predicted labels (float) in range [1, 5] 
            from the Universal Sentence Encoder model
            (see STS-Project/notebooks/embedModel-Dev-Train-Test.ipynb). Defaults to None.
        **kwargs: allows custom weight setting:
            'a' is associated with dep_preds,
            'b' with pawar_preds,
            'c' with embed_preds
    
    Returns:
        list: a list of rounded ensemble predictions (ints) in range [1, 5].
    """
    opts = kwargs.keys()
    ensemble_predics = []

    if embed_preds is not None:
        a = 0.4 if "a" not in opts else kwargs["a"]
        b = 0.1 if "b" not in opts else kwargs["b"]
        c = 0.5 if "c" not in opts else kwargs["c"]

        for dep, pawar, embed in zip(dep_preds, pawar_preds, embed_preds):
            weighting = (dep * a) + (pawar * b) + (embed * c)
            ensemble_predics.append(int(np.round(weighting)))
    else:
        a = 0.95 if "a" not in opts else kwargs["a"]
        b = 0.05 if "b" not in opts else kwargs["b"]

        for dep, pawar in zip(dep_preds, pawar_preds):
            weighting = (dep * a) + (pawar * b)
            ensemble_predics.append(int(np.round(weighting)))

    return ensemble_predics


def main():
    description = (
        "World Domination STS Ensemble Models Head (Ensembles Dependency Tree Model, "
        "(Pawar, 2018) Model, and Universal Sentence Encoder Model)"
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
    parser.add_argument(
        "-n",
        "--no_embed",
        help=(
            f"Drops the Universal Sentence Encoder Model from the ensemble, pulls "
            f"precooked results from: {Path('embeds/*')}"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--test",
        help=(
            f"Runs Ensemble Head on test mode: Runs only the test-set through the"
            f"entire World Domination STS architecture. Produces label predictions "
            f"at: {Path('results/*')}"
        ),
        action="store_true",
    )
    args = parser.parse_args()
    log = not args.quiet

    embed_paths = {
        "dev": "embeds/dev_embed_predics.csv",
        "train": "embeds/train_embed_predics.csv",
        "test": "embeds/test_embed_predics.csv",
    }

    if args.test:
        test = read_data(["test"], args.corpus_path, log)

        test_docs = preprocess_raw(test)
        test_dep_pred = depFit_Predict(test_docs)

        test_disam = disambiguate_pipe(test, "Test")
        test_pawar_pred = pawarFit_Predict(test_disam)

        if not args.no_embed:
            test_embed_pred = list(
                pd.read_csv(embed_paths["test"], index_col=0)["noRound"]
            )

            ensemble_preds = ensemble_head(
                test_dep_pred, test_pawar_pred, test_embed_pred
            )
            tag = "ensemblePredic_ne_final"
        else:
            ensemble_preds = ensemble_head(test_dep_pred, test_pawar_pred)
            tag = "ensemblePredic_final"

        test["prediction"] = ensemble_preds
        tmp = test[["id", "prediction"]]
        write_results(tmp, "test", tag)
        print("\nTest + Predictions DF head:")
        print(test.head())

        if log:
            log_frame(test, name="test", tag=tag)

        return

    else:
        dfs = read_data(["dev", "train"], args.corpus_path, log)
        dev = dfs["dev"]
        train = dfs["train"]

        dev_docs = preprocess_raw(dev)
        train_docs = preprocess_raw(train)

        dev_dep_pred = depFit_Predict(dev_docs)
        train_dep_pred = depFit_Predict(train_docs)

        dev_disam = disambiguate_pipe(dev, "Dev")
        train_disam = disambiguate_pipe(train, "Train")

        dev_pawar_pred = pawarFit_Predict(dev_disam)
        train_pawar_pred = pawarFit_Predict(train_disam)

        if not args.no_embed:
            # Universal Sentence Encoder predictions are "precooked" for project purposes
            #  see embedModel_Dev_Train_Test.ipynb for process details
            dev_embed_pred = list(pd.read_csv(embed_paths["dev"], index_col=0)["noRound"])
            train_embed_pred = list(
                pd.read_csv(embed_paths["train"], index_col=0)["noRound"]
            )

            dev_ensemble_preds = ensemble_head(
                dev_dep_pred, dev_pawar_pred, dev_embed_pred
            )
            train_ensemble_preds = ensemble_head(
                train_dep_pred, train_pawar_pred, train_embed_pred
            )
        else:
            dev_ensemble_preds = ensemble_head(dev_dep_pred, dev_pawar_pred)
            train_ensemble_preds = ensemble_head(train_dep_pred, train_pawar_pred)

        dev["ensemblePredics"] = dev_ensemble_preds
        train["ensemblePredics"] = train_ensemble_preds

        if log:
            for df, name in zip([dev, train], ["dev", "train"]):
                log_frame(df, name=name, tag="ensemble_predics")

        for df, name in zip([dev, train], ["Dev", "Train"]):
            acc = accuracy(df["ensemblePredics"], df["gold"])
            _rmse = rmse(df["ensemblePredics"], df["gold"])
            pear_corr = pearsonr(list(df["ensemblePredics"]), list(df["gold"]))
            cols = ["RMSE", "Accuracy", "Pearson's R", "Pearson's R p-val"]
            vals = [_rmse, acc, pear_corr[0], pear_corr[1]]
            stats = pd.DataFrame(
                list(df["ensemblePredics"]), columns=["Predic_Label"]
            ).describe()
            extra = pd.DataFrame(vals, index=cols, columns=["Predic_Label"])
            print(f"\n{name} Gold stats: ")
            print(pd.DataFrame(list(df["gold"]), columns=["Gold_Label"]).describe().T)
            print(f"\n{name} Ensemble Models Prediction stats: ")
            print(stats.append(extra).T)
            print("\n------")

        for df, name in zip([dev, train], ["Dev", "Train"]):
            print(f"\n{name} Prediction Metrics:")
            metrics = get_scores(list(df["ensemblePredics"]), list(df["gold"]))
            print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
