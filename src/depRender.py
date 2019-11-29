import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy import displacy

from corpusReader import read_data
from enrichPipe import preprocess


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def accuracy(predictions, targets):
    p = 0
    for idx, gold in enumerate(targets):
        predcition = predictions[idx]
        if gold == predcition:
            p += 1
    return float(p) / len(predictions)


# Scraped from evaluation.py
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


nlp = spacy.load("en_core_web_sm")
dev = read_data(["dev"])
# dev = preprocess(dev, name="dev", tag="enriched", log=True)

docs = []
cols = ["s1", "s2"]

for i in range(dev.shape[0]):
    doc = [nlp(dev[col].iloc[i]) for col in cols]
    score = dev["gold"].iloc[i]
    docs.append([doc[0], doc[1], score])

sc0 = []
# Bad weightings:
sc1 = []
sc2 = []
sc3 = []
count = 0
for doc in docs:
    s1 = doc[0]
    s2 = doc[1]
    gold = doc[2]
    s1_root = [token for token in s1 if token.dep_ == "ROOT"][0]
    s2_root = [token for token in s2 if token.dep_ == "ROOT"][0]
    s1_subjects = list(s1_root.lefts) + list(s1_root.rights)
    s2_subjects = list(s2_root.lefts) + list(s2_root.rights)
    s1_subjects = set([token.text for token in s1_subjects] + [s1_root.lemma_])
    s2_subjects = set([token.text for token in s2_subjects] + [s2_root.lemma_])
    overlap = s1_subjects.intersection(s2_subjects)
    total = s1_subjects.union(s2_subjects)
    diff = total.difference(overlap)
    if len(total) == 0:
        # 0 subject overlap counter
        count += 1
        coverage0, coverage1, coverage2, coverage3 = (0, 0, 0, 0)
    else:
        # Weighting pulled out of my ass, think it works out to add 1
        coverage0 = (len(overlap) + len(total)) / (2 * len(total))
        # Alt1. Weighting pulled out of my ass
        coverage1 = (len(overlap) + (0.5 * len(total))) / (2 * len(total))
        # Alt2. weighting pulled out of my ass
        coverage3 = (len(overlap) + len(diff)) / (2 * len(total))
        # Raw
        coverage2 = len(overlap) / len(total)
    coverage = [coverage0, coverage1, coverage2, coverage3]
    pushed = [((val * 4) + 1) for val in coverage]
    sc0.append(pushed[0])
    # Bad weightings:
    sc1.append(pushed[1])
    sc2.append(pushed[2])
    sc3.append(pushed[3])
    # # Debug:
    # print(f"gold: {gold}")
    # print(f"s1 : {s1}")
    # print(f"s2: {s2}")
    # print(f"roots: {s1_root}, {s2_root}")
    # print(f"s1 subjects: {s1_subjects}")
    # print(f"s2 subjects: {s2_subjects}")
    # print(f"complete set: {total}")
    # print(f"overlap: {overlap}, difference: {diff}")
    # print(f"raw scores: {coverage} scaled scores: {np.ceil(pushed)}")
    # print("```")

golds = np.asarray([doc[2] for doc in docs])
sc0 = np.floor(sc0)
# Bad weightings:
sc1 = np.floor(sc1)
sc2 = np.floor(sc2)
sc3 = np.floor(sc3)
print(f"Gold stats: {pd.DataFrame(golds).describe()}")
print(f"\nCoverage0 stats: {pd.DataFrame(sc0).describe()}")
# Bad weightings:
print(f"\nCoverage1 stats: {pd.DataFrame(sc1).describe()}")
print(f"\nCoverage2 stats: {pd.DataFrame(sc2).describe()}")
print(f"\nCoverage3 stats: {pd.DataFrame(sc3).describe()}")

glen = len(golds)
assert glen == len(sc0) and glen == len(sc1) and glen == len(sc2) and glen == len(sc3)

print(f"\ncoverage0 rmse: {rmse(sc0, golds)}, accuracy {accuracy(sc0, golds)}")
# Bad weightings:
print(f"coverage1 rmse: {rmse(sc1, golds)}, accuracy {accuracy(sc1, golds)}")
print(f"coverage2 rmse: {rmse(sc2, golds)}, accuracy {accuracy(sc2, golds)}")
print(f"coverage3 rmse: {rmse(sc3, golds)}, accuracy {accuracy(sc3, golds)}")
print(f"Num sentences w/ no shared nodes in root tree: {count}")
print(json.dumps(get_scores(sc0, golds)))

# LocalHost Serve:
# displacy.serve(docs, style="dep")

# # Save to .svg
# svg = displacy.render(docs, style="dep")
# log_path = Path("log", "dep.svg")
# with log_path.open("w", encoding="utf-8") as dep_log:
#     dep_log.write(svg)
