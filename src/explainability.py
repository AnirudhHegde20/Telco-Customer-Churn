import os
import numpy as np
import shap
import matplotlib.pyplot as plt


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
    """
    Generate and save a SHAP summary plot for a trained pipeline.

    pipeline: fitted sklearn/imb pipeline with steps: preprocess, smote, model
    X_train: original training DataFrame (before preprocessing)
    model_name: label used in filename (e.g. 'logistic_regression')
    model_type: 'linear' or 'tree' (used for explainer choice hint)
    max_samples: limit number of rows for faster SHAP computation
    output_dir: directory to save figures in
    """

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
