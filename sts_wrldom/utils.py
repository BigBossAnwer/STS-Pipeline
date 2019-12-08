import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log_frame(df, name, tag):
    """Logs a dataframe as a .csv to $cwd/log
    
    Args:
        df: dataframe to log
        name (str): the name of the logged .csv
        tag (str): the tag of the logged .csv
    """
    try:
        Path("log").mkdir(exist_ok=True)
        df.to_csv(str(Path("log", name + "_" + tag + ".csv")), index=False)

    except IOError:
        print("Error: Log write failed")
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def write_results(df, name, tag):
    """Writes results (predictions) in the format requested as a .txt in $cwd/results 
    
    Args:
        df: the results dataframe
        name (str): the name of the written .txt
        tag (str): the tag of the written .txt
    """
    try:
        Path("results").mkdir(exist_ok=True)
        df.to_csv(str(Path("results", name + "_" + tag + ".txt")), sep="\t", index=False)

    except IOError:
        print("Error: Log write failed")
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def rmse(predictions, targets):
    """Computes Root Mean Squared Error
    
    Args:
        predictions (list): a list of predicted labels
        targets (list): a list of gold labels
    
    Returns:
        numpy.float64: the RMSE between the predictions and the gold labels
    """
    assert len(predictions) == len(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


def accuracy(predictions, targets):
    """Computes raw accuracy (True Predictions) / (All Predictions)
    
    Args:
        predictions (list): a list of predicted labels
        targets (list): a list of gold labels
    
    Returns:
        float: the raw accuracy between the predictions and the gold labels
    """
    assert len(predictions) == len(targets)
    count_pos = 0
    for predic, gold in zip(predictions, targets):
        if predic == gold:
            count_pos += 1

    return float(count_pos) / len(targets)


def get_scores(predictions, targets, prec=3):
    """Returns a dictionary containing overall and for each label their respective
    recall, precision, and F1 score
    
    Args:
        predictions (list): a list of predicted labels
        targets (list): a list of of gold labels
        prec (int, optional): precision of metric rounding. Defaults to 3.
    
    Returns:
        dict: a dictionary of metrics like:
            {
                "micro": {
                    "recall": float,
                    "precision": float,
                    "fscore": float
                }
                "1": ...
                ...
                "5": ...
            }
    """
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
