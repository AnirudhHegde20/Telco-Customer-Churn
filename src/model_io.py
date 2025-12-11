import os
import json
import joblib


def save_model(pipeline, best_threshold, c_churn, c_offer, model_dir="models"):
    """
    Save the trained pipeline and config (threshold + costs) to disk.
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. Save the pipeline
    model_path = os.path.join(model_dir, "rf_churn_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    # 2. Save configuration for later use (e.g. in app)
    config = {
        "best_threshold": float(best_threshold),
        "c_churn": float(c_churn),
        "c_offer": float(c_offer),
    }

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved config to {config_path}")


def load_model(model_dir="models"):
    """
    Load the trained pipeline and config from disk.
    Returns: (pipeline, config_dict)
    """
    model_path = os.path.join(model_dir, "rf_churn_pipeline.joblib")
    config_path = os.path.join(model_dir, "config.json")

    pipeline = joblib.load(model_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    return pipeline, config
