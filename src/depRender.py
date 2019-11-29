import json
import sys

import numpy as np
import pandas as pd
import spacy
from spacy import displacy

from corpusReader import read_data


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


nlp = spacy.load("en_core_web_sm")
dev = read_data(["dev"])
# dev = preprocess(dev, name="dev", tag="enriched", log=True)

cols = ["s1", "s2"]
docs_s1 = []
docs_s2 = []
golds = []
i = 0
scored = False
# # Inefficient
# for i in range(dev.shape[0]):
#     doc = [nlp(dev[col].iloc[i]) for col in cols]
#     score = dev["gold"].iloc[i]
#     docs.append([doc[0], doc[1], score])
# Efficient:
for col in cols:
    for doc in nlp.pipe(dev[col].values, disable=["tagger"], batch_size=50, n_threads=4):
        if doc.is_parsed:
            if col == "s1":
                docs_s1.append(doc)
            if col == "s2":
                docs_s2.append(doc)
            if not scored:
                golds.append(dev["gold"].iloc[i])
                i += 1
        # Ensure equal length lists regardless of parse
        else:
            if col == "s1":
                docs_s1.append(None)
            if col == "s2":
                docs_s2.append(None)
            if not scored:
                golds.append(None)
                i += 1
    scored = True

score_0 = []
score_1 = []
score_2 = []
score_3 = []
count = 0
for doc in zip(docs_s1, docs_s2, golds):
    s1 = doc[0]
    s2 = doc[1]
    gold = doc[2]
    if s1 is None or s2 is None or gold is None:
        continue
    else:
        s1_root = [token for token in s1 if token.dep_ == "ROOT"][0]
        s2_root = [token for token in s2 if token.dep_ == "ROOT"][0]
        s1_subjects = list(s1_root.lefts) + list(s1_root.rights)
        s2_subjects = list(s2_root.lefts) + list(s2_root.rights)
        s1_subjects = set(
            [token.lemma_ for token in s1_subjects if not token.is_stop]
            + [s1_root.lemma_]
        )
        s2_subjects = set(
            [token.lemma_ for token in s2_subjects if not token.is_stop]
            + [s2_root.lemma_]
        )
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
            # Alt1. weighting pulled out of my ass
            coverage1 = (len(overlap) + (0.5 * len(total))) / (1.5 * len(total))
            # Alt2. weighting pulled out of my ass
            coverage2 = (len(overlap) + len(diff)) / (len(diff) + len(total))
            # Raw
            coverage3 = len(overlap) / len(total)
        coverage = [coverage0, coverage1, coverage2, coverage3]
        scaled = [((val * 4) + 1) for val in coverage]
        score_0.append(scaled[0])
        score_1.append(scaled[1])
        score_2.append(scaled[2])
        score_3.append(scaled[3])
        # # Debug:
        # print(f"gold: {gold}")
        # print(f"s1 : {s1}")
        # print(f"s2: {s2}")
        # print(f"roots: {s1_root}, {s2_root}")
        # print(f"s1 subjects: {s1_subjects}")
        # print(f"s2 subjects: {s2_subjects}")
        # print(f"complete set: {total}")
        # print(f"overlap: {overlap}, difference: {diff}")
        # print(f"raw scores: {coverage} scaled scores: {np.ceil(scaled)}")
        # print("```")

golds = np.asarray(golds)
score_0 = np.floor(score_0)
score_1 = np.floor(score_1)
score_2 = np.floor(score_2)
score_3 = np.floor(score_3)
print("Gold stats: ")
print(pd.DataFrame(golds, columns=["Tags"]).describe().T)
print("\nCoverage0 stats: ")
print(pd.DataFrame(score_0, columns=["Tags"]).describe().T)
print("\nCoverage1 stats: ")
print(pd.DataFrame(score_1, columns=["Tags"]).describe().T)
print("\nCoverage2 stats: ")
print(pd.DataFrame(score_2, columns=["Tags"]).describe().T)
print("\nCoverage3 stats: ")
print(pd.DataFrame(score_3, columns=["Tags"]).describe().T)

gold_len = len(golds)
assert (
    gold_len == len(score_0)
    and gold_len == len(score_1)
    and gold_len == len(score_2)
    and gold_len == len(score_3)
)

print(f"\nCoverage0 rmse: {rmse(score_0, golds)}, accuracy {accuracy(score_0, golds)}")
# Bad weightings:
print(f"Coverage1 rmse: {rmse(score_1, golds)}, accuracy {accuracy(score_1, golds)}")
print(f"Coverage2 rmse: {rmse(score_2, golds)}, accuracy {accuracy(score_2, golds)}")
print(f"Coverage3 rmse: {rmse(score_3, golds)}, accuracy {accuracy(score_3, golds)}")
print(f"# sentences w/ no shared nodes in root tree: {count}")
print("\nCoverage0 metrics:")
print(json.dumps(get_scores(score_0, golds), indent=2))
print("\nCoverage1 metrics:")
print(json.dumps(get_scores(score_1, golds), indent=2))
print("\nCoverage2 metrics:")
print(json.dumps(get_scores(score_2, golds), indent=2))
print("\nCoverage3 metrics:")
print(json.dumps(get_scores(score_3, golds), indent=2))

# # Dependency Parse Tree LocalHost Serve:
# displacy.serve(docs, style="dep")

# # Save to .svg
# svg = displacy.render(docs, style="dep")
# log_path = Path("log", "dep.svg")
# with log_path.open("w", encoding="utf-8") as dep_log:
#     dep_log.write(svg)
