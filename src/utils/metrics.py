import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


def calc_metrics(targets, outputs):
    y_true = torch.argmax(targets, dim=1) if targets.ndim > 1 else targets
    y_pred = torch.argmax(outputs, dim=1) if outputs.ndim > 1 else outputs
    # Calculate confusion matrix values: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Calculate other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = "N/A"  # Handle the case when there are no positive labels

    mcc = matthews_corrcoef(y_true, y_pred)
    # Create a pandas dataframe to store the results
    metrics = {
        "TP": [tp],
        "TN": [tn],
        "FP": [fp],
        "FN": [fn],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-score": [f1],
        "AUC": [auc],
        "MCC": [mcc],
    }
    return pd.DataFrame(metrics)


# Example usage:
# truthset = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
# testset = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
# df = calc_metrics(truthset, testset)
# print(df)
