import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_and_preprocess_data, get_processed_data, TARGET_COLUMN

# Define a path to a small sample of your data for testing
SAMPLE_DATA_PATH = 'data/raw/transactions.csv'


def test_load_and_preprocess_data():
    """Verify that data loads correctly and initial cleanup is performed."""
    df = load_and_preprocess_data(SAMPLE_DATA_PATH)

    # Check if essential columns exist
    assert 'Value' in df.columns
    assert 'CustomerId' in df.columns
    # Check that there are no missing values in critical columns
    assert df['Value'].isnull().sum() == 0
    assert not df.empty


def test_rfm_and_target_generation():
    """Verify that the RFM processor creates the proxy target variable."""
    # We use a smaller subset for faster testing
    final_df, rfm_p, woe_t, scaler = get_processed_data(SAMPLE_DATA_PATH)

    # Check if the proxy target 'is_high_risk' exists
    assert TARGET_COLUMN in final_df.columns
    # Check that it contains only binary values (0 or 1)
    assert set(final_df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_feature_scaling():
    """Verify that numerical features are properly standardized."""
    final_df, _, _, _ = get_processed_data(SAMPLE_DATA_PATH)

    # Standardized features should have a mean very close to 0
    # Checking 'Amount' as a representative feature
    mean_val = final_df['Amount'].mean()
    assert np.isclose(mean_val, 0, atol=1e-1)


def test_data_shape_consistency():
    """Ensure the processed data maintains customer-level granularity."""
    raw_df = pd.read_csv(SAMPLE_DATA_PATH)
    final_df, _, _, _ = get_processed_data(SAMPLE_DATA_PATH)

    # The final dataframe should have one row per unique CustomerId
    assert final_df.shape[0] == raw_df['CustomerId'].nunique()