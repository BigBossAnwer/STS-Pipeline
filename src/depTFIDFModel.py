import json

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from corpusReader import read_data
from enrichPipe import preprocess_raw


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def accuracy(predictions, targets):
    count_pos = 0
    for idx, gold in enumerate(targets):
        if gold == predictions[idx]:
            count_pos += 1

    return float(count_pos) / len(targets)


# Scraped from evaluation.py, returns recall, precision, f1
def get_scores(predictions, targets, prec=3):
    label_set = [1, 2, 3, 4, 5]
    classification_report = {}
    classification_report["micro"] = {"recall": 0.0, "precision": 0.0, "fscore": 0.0}
    for label in label_set:
        classification_report[label] = {"recall": 0.0, "precision": 0.0, "fscore": 0.0}
        tp, fp, fn = 0, 0, 0
        for idx, gold in enumerate(targets):
            prediction = predictions[idx]
            if gold == prediction:
                if prediction == label:
                    tp += 1
            else:
                if prediction == label:
                    fp += 1
                else:
                    fn += 1
        try:
            recall = float(tp) / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        try:
            precision = float(tp) / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            fscore = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            fscore = 0.0
        classification_report[label]["recall"] = round(recall, prec)
        classification_report[label]["precision"] = round(precision, prec)
        classification_report[label]["fscore"] = round(fscore, prec)
        classification_report["micro"]["recall"] += recall
        classification_report["micro"]["precision"] += precision
        classification_report["micro"]["fscore"] += fscore

    for key in classification_report["micro"].keys():
        classification_report["micro"][key] /= len(label_set)
        classification_report["micro"][key] = round(
            classification_report["micro"][key], prec
        )

    return classification_report


# Returns a list of lemmas from a doc of Spacy tokens
def clean_tokens(tokens):
    cleaned_tokens = []
    for token in tokens:
        if not token.is_punct:
            if token.lemma_ == "-PRON-":
                cleaned_tokens.append(token.text)
            else:
                cleaned_tokens.append(token.lemma_)
    return cleaned_tokens


# Returns a 3-tuple (tfidfer, tfidf_mat, average_tfidf):
# tfidfer = the object with access to the fit vocabulary,
# tfidf_mat = the scipy sparse matrix tfidf matrix with (docs, vocab) dimension
# average_tidf = the average tfidf value over all documents, over all words
def tfidf_fit_transform(docs):
    tfidfer = TfidfVectorizer(min_df=0, sublinear_tf=True, lowercase=False)
    tfidf_mat = tfidfer.fit_transform(docs)
    (x, y, z) = scipy.sparse.find(tfidf_mat)
    countings = np.bincount(x)
    sums = np.bincount(x, weights=z)
    averages = sums / countings
    avg_s = 0
    avg_c = 0
    for avg in averages:
        avg_s += avg
        avg_c += 1
    average_tfidf = avg_s / avg_c

    return tfidfer, tfidf_mat, average_tfidf


def main():
    dfs = read_data(["dev", "train"])
    dev_docs = preprocess_raw(dfs["dev"])
    train_docs = preprocess_raw(dfs["train"])
    all_docs = dev_docs + train_docs

    cleaned_docs = []
    for doc_tuple in all_docs:
        # doc_tuple looks like (s1 - spacy processed, s2 - spacy processed, gold)
        for doc in doc_tuple[:2]:
            cleaned_docs.append(" ".join(clean_tokens(doc)))

    tfidfer, tfidf_mat, average_tfidf = tfidf_fit_transform(cleaned_docs)

    predictions = []
    for idx, doc_tuple in enumerate(all_docs):
        s1 = doc_tuple[0]
        s2 = doc_tuple[1]
        gold = doc_tuple[2]
        if s1 is None or s2 is None or gold is None:
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

            # diff = total.difference(overlap)
            # diff_raw = total_raw.difference(overlap_raw)

            # Use full lemmatized sentence TFIDF prediction in cases of "low" dependency tree
            # overlap probability. This"low"/"confidence_threshold" and weighting of
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
            # # Debug:
            # print(f"gold: {gold}")
            # print(f"s1 : {s1}")
            # print(f"s2: {s2}")
            # print(f"complete set: {total} | {total_raw}")
            # print(f"overlap: {overlap} | {overlap_raw}, difference: {diff} | {diff_raw}")
            # print(f"raw prediction: {prediction} scaled prediction: {np.round(scaled)}")
            # print("```")

    golds = np.asarray([doc[2] for doc in all_docs])
    predictions = np.round(predictions)
    print("Gold stats: ")
    print(pd.DataFrame(golds, columns=["Tags"]).describe().T)
    print("\nPrediction stats: ")
    print(pd.DataFrame(predictions, columns=["Tags"]).describe().T)

    gold_len = len(golds)
    assert gold_len == len(predictions)

    print(f"\nPrediction RMSE: {rmse(predictions, golds)}", end="")
    print(f", Accuracy: {accuracy(predictions, golds)}")
    print("\nPrediction metrics:")
    print(json.dumps(get_scores(predictions, golds), indent=2))


if __name__ == "__main__":
    main()

