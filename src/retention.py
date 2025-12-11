import pandas as pd
import numpy as np


def add_risk_scores(customer_ids, y_proba, c_churn=200.0, c_offer=20.0):
    """
    Combine customer IDs with churn probabilities and compute:
    - expected loss if not targeted
    - expected gain if targeted
    """

    df = pd.DataFrame({
        "customerID": customer_ids.values,
        "p_churn": y_proba
    })

    df["expected_loss"] = df["p_churn"] * c_churn
    df["expected_gain_if_targeted"] = (df["p_churn"] * c_churn) - c_offer

    return df.sort_values("expected_loss", ascending=False)


def select_top_customers(df, budget, cost_per_offer=20.0):
    """
    Select customers to target given a budget (in dollars), 
    assuming each offer costs cost_per_offer.

    Example:
    - budget = 20000
    - cost_per_offer = 20
    -> can target 1000 customers
    """

    max_customers = int(budget // cost_per_offer)

    return df.head(max_customers)
