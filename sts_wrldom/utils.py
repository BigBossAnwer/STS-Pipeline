import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log_frame(df, name, tag):
    try:
        Path("log").mkdir(exist_ok=True)
        df.to_csv(str(Path("log", name + "_" + tag + ".csv")))

    except IOError:
        print("Error: Log write failed")
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def write_results(df, name, tag):
    try:
        Path("results").mkdir(exist_ok=True)
        df.to_csv(str(Path("results", name + "_" + tag + ".txt")), sep="\t", index=False)

    except IOError:
        print("Error: Log write failed")
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def rmse(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


def accuracy(predictions, targets):
    assert len(predictions) == len(targets)
    count_pos = 0
    for predic, gold in zip(predictions, targets):
        if predic == gold:
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
