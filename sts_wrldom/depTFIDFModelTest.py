import json

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from sts_wrldom.corpusReader import read_data
from sts_wrldom.depTFIDFModel import clean_tokens, tfidf_fit_transform
from sts_wrldom.enrichPipe import preprocess_raw
from sts_wrldom.utils import accuracy, get_scores, log_frame, rmse


def main():
    """Parameter testing for Dependency Tree + TFIDF Weights Model"""
    dfs = read_data(["dev", "train"])
    dev = dfs["dev"]
    train = dfs["train"]
    dev_train = dev.append(train)
    dev_docs = preprocess_raw(dev)
    train_docs = preprocess_raw(train)
    all_docs = dev_docs + train_docs

    cleaned_docs = []
    for doc_tuple in all_docs:
        for doc in doc_tuple:
            cleaned_docs.append(" ".join(clean_tokens(doc)))

    tfidfer, tfidf_mat, average_tfidf = tfidf_fit_transform(cleaned_docs)

    collated = []
    for idx, doc_tuple in enumerate(all_docs):
        s1 = doc_tuple[0]
        s2 = doc_tuple[1]
        if s1 is None or s2 is None:
            print(f"Bad row: {idx, doc_tuple}")
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

            diff = total.difference(overlap)
            diff_raw = total_raw.difference(overlap_raw)

            collated.append(
                [
                    overlap,
                    overlap_raw,
                    overlap_raw_score,
                    total,
                    total_raw,
                    total_raw_score,
                    diff,
                    diff_raw,
                ]
            )

    for weighting_test in np.arange(0.2, 0.31, 0.01):
        covs = {"cov0": [], "cov1": [], "cov2": []}
        for confidence_test in np.arange(0.6, 0.9, 0.05):
            score_0 = []
            score_1 = []
            score_2 = []
            count = 0

            for row in collated:
                (
                    overlap,
                    overlap_raw,
                    overlap_raw_score,
                    total,
                    total_raw,
                    total_raw_score,
                    diff,
                    diff_raw,
                ) = row

                if len(overlap) / len(total) <= confidence_test:
                    coverage0 = len(overlap) / len(total)
                    coverage1 = min(overlap_raw_score / total_raw_score, 1)
                else:
                    coverage0 = (len(overlap) + len(total)) / (2 * len(total))
                    coverage1 = (
                        (len(overlap) + len(total)) / (2 * len(total)) * (weighting_test)
                    ) + (
                        min(overlap_raw_score / total_raw_score, 1) * (1 - weighting_test)
                    )

                if len(overlap_raw) / len(total_raw) <= confidence_test:
                    count += 1
                    coverage2 = (
                        ((len(overlap_raw)) / (len(total_raw))) * (weighting_test)
                    ) + (
                        min(overlap_raw_score / total_raw_score, 1) * (1 - weighting_test)
                    )
                else:
                    coverage2 = (
                        ((len(overlap_raw) + len(total_raw)) / (2 * len(total_raw)))
                        * (weighting_test)
                    ) + (
                        min(overlap_raw_score / total_raw_score, 1) * (1 - weighting_test)
                    )

                coverage = [coverage0, coverage1, coverage2]
                scaled = [((val * 4) + 1) for val in coverage]
                score_0.append(scaled[0])
                score_1.append(scaled[1])
                score_2.append(scaled[2])

            golds = np.asarray(dev_train["gold"])
            score_0 = np.round(score_0)
            score_1 = np.round(score_1)
            score_2 = np.round(score_2)

            gold_len = len(golds)
            assert gold_len == len(score_0)
            assert gold_len == len(score_1)
            assert gold_len == len(score_2)

            for score, cov in zip([score_0, score_1, score_2], covs.keys()):
                covs[cov].append(
                    [
                        confidence_test,
                        np.mean(score),
                        np.std(score),
                        accuracy(score, golds),
                        count,
                    ]
                )

        print(weighting_test)
        for cov in covs.keys():
            df_temp = pd.DataFrame(
                covs[cov], columns=["test", "mean", "std", "accuracy", "count"]
            )
            if cov == "cov1":
                print(cov)
                print(df_temp)
                log_frame(df_temp, name=cov, tag="paramTest")
            else:
                log_frame(df_temp, name=cov, tag="paramTest")
        print()


if __name__ == "__main__":
    main()
