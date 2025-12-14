import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_processing import get_processed_data, TARGET_COLUMN
import joblib  # for saving transformers

# Configuration
DATA_PATH = '../data/raw/transactions.csv'
RANDOM_STATE = 42
MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'  # Use a local database for tracking
MLFLOW_EXPERIMENT_NAME = 'Credit_Risk_Model_RFM'

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def evaluate_model(y_true, y_pred, y_proba):
    """Calculates and returns model evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }


def train_and_track_model(name, model, X_train, y_train, X_test, y_test, params=None):
    """Trains a model, evaluates it, and logs the results to MLflow."""
    with mlflow.start_run(run_name=name) as run:
        print(f"Starting MLflow run for {name}...")

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Train model
        model.fit(X_train, y_train)

        # Predict and Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)

        # Log metrics
        mlflow.log_metrics(metrics)
        print(f"Metrics for {name}: {metrics}")

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Register the model if AUC is above a threshold (simple check for "best")
        if metrics['roc_auc'] > 0.65:  # A reasonable threshold for a new model
            mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                "CreditRiskModel"  # Model name in the registry
            )
            print(f"Registered model {name} with AUC: {metrics['roc_auc']:.4f}")

        return run.info.run_id, metrics['roc_auc']


def hyperparameter_tuning(model, param_grid, X_train, y_train, name):
    """Performs GridSearchCV and returns the best model and parameters."""
    print(f"Starting GridSearchCV for {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best AUC for {name}: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    # 1. Load and Process Data
    processed_df, rfm_processor, woe_transformer, scaler = get_processed_data(DATA_PATH)

    X = processed_df.drop(columns=[TARGET_COLUMN])
    y = processed_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    # Save the transformers for deployment
    joblib.dump(rfm_processor, '../data/processed/rfm_processor.pkl')
    joblib.dump(woe_transformer, '../data/processed/woe_transformer.pkl')
    joblib.dump(scaler, '../data/processed/scaler.pkl')

    # Drop the temporary index columns used in data_processing for the actual model training
    X_train = X_train.drop(columns=['CustomerId'], errors='ignore')
    X_test = X_test.drop(columns=['CustomerId'], errors='ignore')

    # 2. Model Selection and Training (Task 5)

    # --- Logistic Regression ---
    lr_model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
    lr_params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}

    best_lr, best_lr_params = hyperparameter_tuning(lr_model, lr_params, X_train, y_train, 'LogisticRegression')
    train_and_track_model('Best_LogisticRegression', best_lr, X_train, y_train, X_test, y_test, best_lr_params)

    # --- Random Forest ---
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

    best_rf, best_rf_params = hyperparameter_tuning(rf_model, rf_params, X_train, y_train, 'RandomForest')
    run_id, rf_auc = train_and_track_model('Best_RandomForest', best_rf, X_train, y_train, X_test, y_test,
                                           best_rf_params)

    # Print next steps
    print("\n--- Model Training Complete ---")
    print(f"To view the experiments, run 'mlflow ui' in your terminal and navigate to: {MLFLOW_TRACKING_URI}")
    print(f"The best model (RandomForest, Run ID: {run_id}) has been registered in the MLflow Model Registry.")


if __name__ == '__main__':
    # You must run this script from the project root (credit-risk-model/)
    # for the relative paths to work correctly.
    main()