from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np

scores_dict = {
    "accuracy_score":accuracy_score,
    "f1_score":f1_score,
    "precision_score":precision_score,
    "recall_score":recall_score
}

def compute_metrics (y_hat, y_true):
    metrics = {}
    values = np.unique(y_true)
    
    for i in values:
        metrics[i] = {}
        y_true_ = (y_true == i).astype("int")
        y_hat_ = (y_hat == i).astype("int")

        for score_name, score_fn in scores_dict.items():
            metrics[i][score_name] = score_fn(y_true_, y_hat_)

    metrics = pd.DataFrame(metrics).T
    
    return metrics