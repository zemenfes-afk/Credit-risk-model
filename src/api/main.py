import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionOutput
from src.data_processing import RFM_FEATURES, CUSTOMER_ID, TARGET_COLUMN
import joblib  # for loading transformers

# Configuration
MLFLOW_MODEL_NAME = "CreditRiskModel"
MLFLOW_MODEL_VERSION = 1  # Update this as you register new versions
TRANSFORMER_PATH = '../data/processed/'

# Load the model and transformers
try:
    # Load model from MLflow Registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")

    # Load transformers
    rfm_processor = joblib.load(TRANSFORMER_PATH + 'rfm_processor.pkl')
    woe_transformer = joblib.load(TRANSFORMER_PATH + 'woe_transformer.pkl')
    scaler = joblib.load(TRANSFORMER_PATH + 'scaler.pkl')

except Exception as e:
    print(f"Error loading MLflow model or transformers: {e}")
    # Raise an exception to stop the service if loading fails
    raise RuntimeError("Failed to load necessary model components.") from e

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predicts credit risk probability for Buy-Now-Pay-Later customers using RFM-based features."
)


def transform_input_data(df_raw: pd.DataFrame):
    """Applies the RFM, WoE, and Scaling transformations to raw input data."""

    # 1. RFM and Target Creation (uses the fitted RFM processor)
    # Note: Target will be recalculated but is not used in prediction
    df_rfm_target = rfm_processor.transform(df_raw)
    customer_id = df_rfm_target[CUSTOMER_ID].iloc[0]

    # 2. WoE Transformation (using the fitted WoE transformer)
    # Features to process (excluding Recency/Frequency/Monetary for clean WoE/Scaling)
    COLUMNS_TO_DROP = [TARGET_COLUMN, CUSTOMER_ID]
    X_woe_in = df_rfm_target.drop(columns=COLUMNS_TO_DROP)

    # The WoE transformer is assumed to handle the categorical features in X_woe_in
    X_transformed = woe_transformer.transform(X_woe_in)

    # 3. Standardization (using the fitted scaler)
    NUMERICAL_FEATURES_FINAL = [col for col in X_transformed.columns if
                                col not in [f for f in X_transformed.columns if '_WoE' in f]]
    X_transformed[NUMERICAL_FEATURES_FINAL] = scaler.transform(X_transformed[NUMERICAL_FEATURES_FINAL])

    return X_transformed, customer_id


def calculate_credit_score(risk_prob: float) -> int:
    """Simple linear mapping of risk probability to a credit score (300-850)."""
    # Assuming a linear inverse relationship: Low probability -> High score
    # Score = 850 - (risk_prob * 550)
    score = 850 - (risk_prob * (850 - 300))
    return max(300, min(850, int(score)))


@app.get("/")
def health_check():
    return {"status": "ok", "model": MLFLOW_MODEL_NAME}


@app.post("/predict", response_model=PredictionOutput)
def predict_risk(data: CustomerData):
    """
    Accepts a list of a customer's transactions and returns a credit risk prediction.
    """
    if not data.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    # Convert Pydantic models to a DataFrame
    raw_transactions = [t.dict() for t in data.transactions]
    df_raw = pd.DataFrame(raw_transactions)

    try:
        # Pre-process the data
        X_final, customer_id = transform_input_data(df_raw)

        # Make prediction
        # The model expects only one row of customer data (post-RFM processing)
        risk_probability = model.predict_proba(X_final.iloc[[0]])[:, 1][0]

        # Calculate Credit Score (Task 4 requirement)
        credit_score = calculate_credit_score(risk_probability)

        # Determine risk category
        if risk_probability > 0.6:
            risk_category = "High Risk"
        elif risk_probability > 0.3:
            risk_category = "Medium Risk"
        else:
            risk_category = "Low Risk"

        return PredictionOutput(
            CustomerId=customer_id,
            RiskProbability=float(risk_probability),
            CreditScore=credit_score,
            RiskCategory=risk_category
        )

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")