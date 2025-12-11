import numpy as np
from sklearn.metrics import confusion_matrix


def compute_profit_curve(y_true, y_proba, c_churn=200.0, c_offer=20.0):
    """
    For a range of thresholds between 0 and 1:
    - Convert predicted probabilities into churn / no-churn
    - Compute confusion matrix
    - Compute expected profit

    Returns: list of dicts, one per threshold:
    [
        {
            "threshold": t,
            "profit": ...,
            "tp": ...,
            "fp": ...,
            "fn": ...,
            "tn": ...,
            "recall": ...,
            "precision": ...
        },
        ...
    ]
    """

    thresholds = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00
    results = []

    for t in thresholds:
        # turn probabilities into predicted labels
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Profit logic:
        # - TP: customer would churn, we offer, they stay
        #       We avoid losing C_churn but pay C_offer  -> + (C_churn - C_offer)
        # - FP: customer wouldn't churn, but we still offer
        #       No benefit, but we pay C_offer           -> - C_offer
        # - FN: customer churns and we didn't catch them
        #       We lose C_churn                          -> - C_churn
        # - TN: no churn and no offer                    -> 0

        profit = (
            tp * (c_churn - c_offer)
            - fp * c_offer
            - fn * c_churn
        )

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results.append(
            {
                "threshold": float(t),
                "profit": float(profit),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "recall": float(recall),
                "precision": float(precision),
            }
        )

    return results


def find_best_threshold(results):
    """
    Given the list from compute_profit_curve, return the dict
    with the highest profit.
    """
    best = max(results, key=lambda d: d["profit"])
    return best
