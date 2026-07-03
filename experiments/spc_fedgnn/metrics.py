import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def summarize_client_metrics(client_rows):
    accs = np.array([row["accuracy"] for row in client_rows], dtype=float)
    f1s = np.array([row["macro_f1"] for row in client_rows], dtype=float)
    return {
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "worst_accuracy": float(accs.min()),
        "mean_macro_f1": float(f1s.mean()),
        "std_macro_f1": float(f1s.std()),
        "worst_macro_f1": float(f1s.min()),
    }
