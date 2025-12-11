import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_prep import load_data, clean_data
from src.features import build_preprocessor
from src.modeling import build_logistic_pipeline, build_rf_pipeline
from src.evaluation import evaluate_model
from src.profit import compute_profit_curve, find_best_threshold
from src.retention import add_risk_scores, select_top_customers
from src.explainability import generate_shap_summary
from src.explainability import generate_shap_summary, explain_single_customer
from src.model_io import save_model




def run():
    # 1. Load and clean data
    df = load_data("data/Telco-Customer-Churn.csv")
    df = clean_data(df)

    customer_ids = df["customerID"]

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Remove customerID from features
    X = X.drop(columns=["customerID"])

    # 2. Split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Preprocessor
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 4. Logistic Regression
    log_pipe = build_logistic_pipeline(preprocessor)
    log_pipe.fit(X_train, y_train)

    y_pred_log = log_pipe.predict(X_test)
    y_proba_log = log_pipe.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred_log, y_proba_log, model_name="Logistic Regression")

    # 5. SHAP for Logistic Regression
    generate_shap_summary(
        pipeline=log_pipe,
        X_train=X_train,
        model_name="logistic_regression",
        model_type="linear"
    )

    # 6. Random Forest
    rf_pipe = build_rf_pipeline(preprocessor)
    rf_pipe.fit(X_train, y_train)

    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred_rf, y_proba_rf, model_name="Random Forest")

    # 7. SHAP for Random Forest
    generate_shap_summary(
        pipeline=rf_pipe,
        X_train=X_train,
        model_name="random_forest",
        model_type="tree"
    )

    # 8. Profit Optimization
    results = compute_profit_curve(
        y_true=y_test,
        y_proba=y_proba_rf,
        c_churn=200.0,
        c_offer=20.0
    )

    best = find_best_threshold(results)
    print("\n====== Profit-based Threshold (Random Forest) ======")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Max profit: {best['profit']:.2f}")
    print(f"Recall: {best['recall']:.3f}")
    print(f"Precision: {best['precision']:.3f}")

    # 8. Save production model (Random Forest) + config
    save_model(
        pipeline=rf_pipe,
        best_threshold=best["threshold"],
        c_churn=200.0,
        c_offer=20.0,
        model_dir="models"
    )


    # 10. Retention Engine
    risk_df = add_risk_scores(
        customer_ids=ids_test,
        y_proba=y_proba_rf,
        c_churn=200.0,
        c_offer=20.0
    )

    selected = select_top_customers(risk_df, budget=20000)

    print("\n====== Retention Recommendation ======")
    print(selected.head(10))
    print(f"\nTotal customers targeted: {len(selected)}")

    # 11. Local SHAP explanation for top high-risk customer (Random Forest)
    top_customer = selected.iloc[0]
    top_customer_id = top_customer["customerID"]

    # Find this customer's row in X_test using ids_test
    mask = ids_test == top_customer_id
    x_row = X_test[mask].iloc[0]

    rf_explanation = explain_single_customer(
        pipeline=rf_pipe,
        X_train=X_train,
        x_row=x_row,
        model_name="random_forest",
        top_n=5
    )

    print(f"\n====== Local Explanation (Random Forest) for customer {top_customer_id} ======")
    print("Top contributing features (feature, shap_value):")
    for item in rf_explanation:
        direction = "increases churn risk" if item["shap_value"] > 0 else "decreases churn risk"
        print(f"  {item['feature']}: {item['shap_value']:.3f} → {direction}")



if __name__ == "__main__":
    run()
