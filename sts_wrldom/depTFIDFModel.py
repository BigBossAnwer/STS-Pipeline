import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer

from sts_wrldom.corpusReader import read_data
from sts_wrldom.enrichPipe import preprocess_raw
from sts_wrldom.utils import accuracy, get_scores, log_frame, rmse


def clean_tokens(tokens):
    """Returns a list of lemmas from a list of spaCy tokens
    
    Args:
        tokens (list(spaCy_token)): the list of spaCy tokens to pull lemmas from
    
    Returns:
        list: a list of lemmas corresponding to the tokens passed in
    """
    cleaned_tokens = []
    for token in tokens:
        if not token.is_punct:
            if token.lemma_ == "-PRON-":
                cleaned_tokens.append(token.text)
            else:
                cleaned_tokens.append(token.lemma_)
    return cleaned_tokens


def tfidf_fit_transform(docs):
    """Returns a 3-tuple containing the sklearn TfidfVectorizer object, the associated
    TFIDF matrix and the average TFIDF value over all non-zero TFIDF values
    
    Args:
        docs (list): a list of str, where every element is the plain text document, 
            TFIDF values are computed on this corpus
    
    Returns:
        tuple: a 3-tuple like
            tuple[0] (sklearn.feature_extraction.text.TfidfVectorizer): object used to 
                compute TFIDF values for the corpus
            tuple[1] (scipy.sparse.csr.csr_matrix): TFIDF matrix with shape (docs, vocab)
            tuple[2] (numpy.float64): the average non-zero TFIDF value
    """
    tfidfer = TfidfVectorizer(min_df=0, sublinear_tf=True, lowercase=False)
    tfidf_mat = tfidfer.fit_transform(docs)
    (x, _, z) = scipy.sparse.find(tfidf_mat)
    countings = np.bincount(x)
    sums = np.bincount(x, weights=z)
    average_tfidf = np.mean(sums / countings)

    return tfidfer, tfidf_mat, average_tfidf


def depFit_Predict(docs):
    """Returns a list of predicted labels using sts_wrldom's Dependency Tree + TFIDF
    weighted bag of words IoU model
    
    Args:
        docs (list): a list of (s1, s2) spaCy processed doc tuples
    
    Returns:
        list: a list of float prediction labels in range [1, 5]
    """
    cleaned_docs = []
    for doc_tuple in docs:
        # doc_tuple looks like (s1 - spacy processed, s2 - spacy processed)
        for doc in doc_tuple:
            cleaned_docs.append(" ".join(clean_tokens(doc)))

    tfidfer, tfidf_mat, average_tfidf = tfidf_fit_transform(cleaned_docs)

    predictions = []
    for idx, doc_tuple in enumerate(docs):
        s1 = doc_tuple[0]
        s2 = doc_tuple[1]
        # gold = doc_tuple[2]
        if s1 is None or s2 is None:
            print(f"Bad row {idx, doc_tuple}")
        else:
            s1_root = [token for token in s1 if token.dep_ == "ROOT"][0]
            s2_root = [token for token in s2 if token.dep_ == "ROOT"][0]
            s1_subjects = list(s1_root.lefts) + list(s1_root.rights)
            s2_subjects = list(s2_root.lefts) + list(s2_root.rights)
            s1_ents = [e.label_ for e in s1.ents]
            s2_ents = [e.label_ for e in s2.ents]

            cleaned_s1_subjects = clean_tokens(s1_subjects)
            cleaned_s2_subjects = clean_tokens(s2_subjects)
            s1_cover = set(cleaned_s1_subjects + [s1_root.lemma_] + s1_ents)
            s2_cover = set(cleaned_s2_subjects + [s2_root.lemma_] + s2_ents)

            cleaned_s1_raw = clean_tokens(s1)
            cleaned_s2_raw = clean_tokens(s2)
            s1_raw_cover = set(cleaned_s1_raw)
            s2_raw_cover = set(cleaned_s2_raw)

            overlap = s1_cover.intersection(s2_cover)
            # TFIDF weighting on raw text (lemmas), not on subjects + roots + entities
            overlap_raw = s1_raw_cover.intersection(s2_raw_cover)
            overlap_raw_tfidf = []
            for elem in overlap_raw:
                if elem in tfidfer.vocabulary_.keys():
                    avg = (
                        tfidf_mat[(idx * 2), tfidfer.vocabulary_[elem]]
                        + tfidf_mat[(idx * 2 + 1), tfidfer.vocabulary_[elem]]
                    ) / 2
                    overlap_raw_tfidf.append(avg)
                else:
                    overlap_raw_tfidf.append(average_tfidf)
            overlap_raw_score = np.sum(overlap_raw_tfidf)

            total = s1_cover.union(s2_cover)
            # TFIDF weighting on raw text (lemmas), not on subjects + roots + entities
            total_raw = s1_raw_cover.union(s1_raw_cover)
            total_raw_tfidf = []
            for elem in total_raw:
                if elem in tfidfer.vocabulary_.keys():
                    avg = (
                        tfidf_mat[(idx * 2), tfidfer.vocabulary_[elem]]
                        + tfidf_mat[(idx * 2 + 1), tfidfer.vocabulary_[elem]]
                    ) / 2
                    total_raw_tfidf.append(avg)
                else:
                    total_raw_tfidf.append(average_tfidf)
            total_raw_score = np.sum(total_raw_tfidf)

            # Use full lemmatized sentence TFIDF prediction in cases of "low" dependency
            # tree overlap probability. This "low"/"confidence_threshold" and weighting of
            # models in confident cases defined by hyper-parameter testing on the
            # dev + train set in depParamTest.py
            confidence_threshold = 0.67
            weighting = 0.23
            if len(overlap) / len(total) <= confidence_threshold:
                prediction = min(overlap_raw_score / total_raw_score, 1)
            else:
                prediction = (
                    (len(overlap) + len(total)) / (2 * len(total)) * (weighting)
                ) + (min(overlap_raw_score / total_raw_score, 1) * (1 - weighting))

            scaled = (prediction * 4) + 1
            predictions.append(scaled)

    return predictions


def main():
    description = (
        "TFIDF Weighted Dependency Tree Coverage STS Model. Running this "
        "standalone amounts to testing"
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

    dfs = read_data(["dev", "train"], args.corpus_path, log)
    dev = dfs["dev"]
    train = dfs["train"]
    dev_train = dev.append(train)

    dev_docs = preprocess_raw(dfs["dev"])
    train_docs = preprocess_raw(dfs["train"])
    dev_train_docs = dev_docs + train_docs

    dev_predics = depFit_Predict(dev_docs)
    train_predics = depFit_Predict(train_docs)
    dev_train_predics = depFit_Predict(dev_train_docs)

    dev["depPredics"] = [int(elem) for elem in np.round(dev_predics)]
    train["depPredics"] = [int(elem) for elem in np.round(train_predics)]
    dev_train["depPredics"] = [int(elem) for elem in np.round(dev_train_predics)]

    if log:
        for df, name in zip([dev, train], ["dev", "train"]):
            log_frame(df, name=name, tag="depTFIDF_predics")

    for df, name in zip([dev, train, dev_train], ["Dev", "Train", "Dev-Train"]):
        acc = accuracy(df["depPredics"], df["gold"])
        _rmse = rmse(df["depPredics"], df["gold"])
        pear_corr = pearsonr(list(df["depPredics"]), list(df["gold"]))
        cols = ["RMSE", "Accuracy", "Pearson's R", "Pearson's R p-val"]
        vals = [_rmse, acc, pear_corr[0], pear_corr[1]]
        stats = pd.DataFrame(list(df["depPredics"]), columns=["Predic_Label"]).describe()
        extra = pd.DataFrame(vals, index=cols, columns=["Predic_Label"])
        print(f"\n{name} Gold stats: ")
        print(pd.DataFrame(list(df["gold"]), columns=["Gold_Label"]).describe().T)
        print(f"\n{name} depTFIDF Model Prediction stats: ")
        print(stats.append(extra).T)
        print("\n------")

    for df, name in zip([dev, train, dev_train], ["Dev", "Train", "Dev-Train"]):
        print(f"\n{name} Prediction Metrics:")
        metrics = get_scores(list(df["depPredics"]), list(df["gold"]))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
