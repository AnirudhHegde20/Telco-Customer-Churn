import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd


def _get_transformed_data_and_feature_names(pipeline, X_sample):
    """
    Helper:
    - Use the pipeline's preprocessor to transform X_sample
    - Extract human-readable feature names for SHAP plots
    """

    preprocessor = pipeline.named_steps["preprocess"]

    # Transform
    X_trans = preprocessor.transform(X_sample)

    # Get feature names from ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Older sklearn versions might not have this;
        # fallback to generic names
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    return X_trans, feature_names


def generate_shap_summary(
    pipeline,
    X_train,
    model_name: str,
    model_type: str,
    max_samples: int = 1000,
    output_dir: str = "reports"
):

    os.makedirs(output_dir, exist_ok=True)

    # 1) Sample a subset for speed
    if len(X_train) > max_samples:
        X_sample = X_train.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_train.copy()

    # 2) Transform and get feature names
    X_trans, feature_names = _get_transformed_data_and_feature_names(pipeline, X_sample)

    # 3) Extract the trained model from the pipeline
    model = pipeline.named_steps["model"]

    # 4) Build SHAP explainer
    # shap.Explainer auto-detects model type; model_type is more for doc clarity
    explainer = shap.Explainer(model, X_trans)

    # 5) Compute SHAP values
    shap_values = explainer(X_trans)

    # 6) Plot summary
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=False,
        max_display=20
    )

    out_path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
    plt.title(f"SHAP Summary - {model_name}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot for {model_name} to: {out_path}")

def explain_single_customer(
    pipeline,
    X_train,
    x_row,
    model_name: str,
    max_samples: int = 1000,
    top_n: int = 5
):
    

    # 1) Build a small background sample for SHAP
    if len(X_train) > max_samples - 1:
        X_bg = X_train.sample(n=max_samples - 1, random_state=42)
    else:
        X_bg = X_train.copy()

    # Append the target row at the end
    X_bg = (
        X_bg.append(x_row)
        if hasattr(X_bg, "append")
        else pd.concat([X_bg, x_row.to_frame().T], axis=0)
    )

    # 2) Transform with the pipeline's preprocessor
    X_trans, feature_names = _get_transformed_data_and_feature_names(
        pipeline, X_bg
    )

    # The last row corresponds to our target customer
    x_trans_target = X_trans[-1, :].reshape(1, -1)

    model = pipeline.named_steps["model"]

    # 3) Build SHAP explainer using background data
    explainer = shap.Explainer(model, X_trans)

    # 4) Get SHAP values for all rows, then pick the last one
    shap_values = explainer(X_trans)

    # shap_values.values can have shapes like:
    # (n_samples, n_features) or (n_samples, 1, n_features)
    vals = shap_values.values
    row_shap = vals[-1]

    # If there is an extra dimension (e.g. (1, n_features)), squeeze it
    if row_shap.ndim > 1:
        row_shap = row_shap[0]

    # 5) Sort features by absolute SHAP impact
    idx_sorted = np.argsort(np.abs(row_shap))[::-1][:top_n]


    explanation = []
    for i in idx_sorted:
        explanation.append(
            {
                "feature": feature_names[i],
                "shap_value": float(row_shap[i]),
            }
        )

    return explanation
