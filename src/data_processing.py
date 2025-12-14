import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xverse.transformer import WoE

# Global Constants
CUSTOMER_ID = 'CustomerId'
TRANSACTION_START_TIME = 'TransactionStartTime'
AMOUNT_COLUMN = 'Amount'
VALUE_COLUMN = 'Value'
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'is_high_risk'


class RFM_Processor(BaseEstimator, TransformerMixin):
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics and
    uses K-Means to cluster customers and assign a 'high-risk' proxy label.
    """

    def __init__(self, snapshot_date_offset_days=1):
        # Define a snapshot date for Recency calculation: the day after the latest transaction
        self.snapshot_date_offset_days = snapshot_date_offset_days
        self.snapshot_date = None
        self.kmeans_model = None
        self.high_risk_cluster = None
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[TRANSACTION_START_TIME] = pd.to_datetime(X_copy[TRANSACTION_START_TIME])

        # 1. Define Snapshot Date
        self.snapshot_date = X_copy[TRANSACTION_START_TIME].max() + pd.Timedelta(days=self.snapshot_date_offset_days)

        # 2. Calculate RFM Metrics
        rfm = X_copy.groupby(CUSTOMER_ID).agg(
            Recency=(TRANSACTION_START_TIME, lambda x: (self.snapshot_date - x.max()).days),
            Frequency=(CUSTOMER_ID, 'size'),
            Monetary=(VALUE_COLUMN, 'sum')
        ).reset_index()

        # Pre-process RFM features
        rfm_scaled = self.scaler.fit_transform(rfm[RFM_FEATURES])

        # 3. Cluster Customers (K-Means)
        # Using 3 clusters as per Task 4 instruction
        self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init='auto')
        rfm['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled)

        # 4. Define and Assign the "High-Risk" Label
        # High-risk = least engaged (high Recency, low Frequency, low Monetary)
        # We find the cluster with the highest average Recency AND lowest avg Frequency/Monetary
        cluster_means = rfm.groupby('Cluster')[RFM_FEATURES].mean()

        # The 'high-risk' cluster is often the one with the highest Recency
        # and lowest Frequency/Monetary. Here, we prioritize high Recency.
        self.high_risk_cluster = cluster_means['Recency'].idxmax()

        return self

    def transform(self, X):
        if self.snapshot_date is None or self.kmeans_model is None:
            raise RuntimeError("RFM_Processor must be fitted before calling transform.")

        X_copy = X.copy()
        X_copy[TRANSACTION_START_TIME] = pd.to_datetime(X_copy[TRANSACTION_START_TIME])

        # Recalculate RFM for the input data
        rfm = X_copy.groupby(CUSTOMER_ID).agg(
            Recency=(TRANSACTION_START_TIME, lambda x: (self.snapshot_date - x.max()).days),
            Frequency=(CUSTOMER_ID, 'size'),
            Monetary=(VALUE_COLUMN, 'sum')
        ).reset_index()

        # Apply the fitted scaler and kmeans model
        rfm_scaled = self.scaler.transform(rfm[RFM_FEATURES])
        rfm['Cluster'] = self.kmeans_model.predict(rfm_scaled)

        # Assign the target variable
        rfm[TARGET_COLUMN] = (rfm['Cluster'] == self.high_risk_cluster).astype(int)

        # Merge the target back into the main DataFrame (dropping duplicate CustomerIds)
        X_processed = X_copy.drop_duplicates(subset=[CUSTOMER_ID], keep='first').copy()
        X_processed = X_processed.merge(
            rfm[[CUSTOMER_ID, 'Recency', 'Frequency', 'Monetary', TARGET_COLUMN]],
            on=CUSTOMER_ID,
            how='left'
        )

        # Drop rows where target variable might be missing (shouldn't happen if all CustomerIds exist)
        X_processed.dropna(subset=[TARGET_COLUMN], inplace=True)

        # Add a placeholder for time-based features (only needed once per customer)
        X_processed['TransactionHour'] = pd.to_datetime(X_processed[TRANSACTION_START_TIME]).dt.hour

        # Select the features for the final model
        feature_cols = [
            'Recency', 'Frequency', 'Monetary', 'Amount', 'Value', 'TransactionHour',
            'ProductId', 'ProductCategory', 'ChannelId', 'ProviderId',
            'CountryCode', 'CurrencyCode', 'PricingStrategy'
        ]

        # Drop features not needed for the final model (like TransactionId, BatchId, etc.)
        # and ensure only one row per customer is kept
        X_final = X_processed[feature_cols + [TARGET_COLUMN, CUSTOMER_ID]].drop_duplicates(subset=[CUSTOMER_ID])

        return X_final


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) transformation to all categorical features.
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.woe_iv = WoE()

    def fit(self, X, y):
        # WoE transformer expects a target variable 'y'
        self.woe_iv.fit(X[self.categorical_features], y)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Transform the specified categorical features
        X_transformed = self.woe_iv.transform(X_copy[self.categorical_features])

        # Replace original categorical columns with WoE columns
        for col in self.categorical_features:
            woe_col = col + '_WoE'
            if woe_col in X_transformed.columns:
                X_copy[col] = X_transformed[woe_col]
            else:
                # If WoE could not be calculated (e.g., all categories have 0 or 1 target),
                # use a fallback like one-hot or label encoding (simplifying to Label Encoding here)
                print(f"Warning: WoE not calculated for {col}. Applying Label Encoding fallback.")
                X_copy[col] = X_copy[col].astype('category').cat.codes

        # Drop columns not needed for the final model (only keeping the features that were transformed)
        # Note: In a real scenario, you'd integrate WoE with the full pipeline.
        # Here we just return the transformed features and the numerical ones.
        numerical_features = X_copy.select_dtypes(include=np.number).columns.tolist()
        return X_copy[numerical_features]


def load_and_preprocess_data(file_path):
    """Loads and performs initial data cleaning."""
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Simple cleanup: Ensure value is positive and handle missingness
    df['Value'] = df['Amount'].abs()
    df.dropna(subset=[CUSTOMER_ID, TRANSACTION_START_TIME, 'Value'], inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def get_processed_data(file_path):
    """Runs the full data processing pipeline."""
    df_raw = load_and_preprocess_data(file_path)

    # Features to be used in the model
    # Note: 'ProductId' might have too many unique values for WoE/OHE,
    # but we include it for completeness as requested.
    ALL_FEATURES = [
        'Amount', 'Value', 'TransactionHour', 'ProductId', 'ProductCategory',
        'ChannelId', 'ProviderId', 'CountryCode', 'CurrencyCode', 'PricingStrategy'
    ]

    CATEGORICAL_FEATURES = [
        'ProductId', 'ProductCategory', 'ChannelId', 'ProviderId',
        'CountryCode', 'CurrencyCode', 'PricingStrategy'
    ]

    NUMERICAL_FEATURES = ['Amount', 'Value', 'TransactionHour']

    # 1. RFM and Target Creation (Task 4)
    rfm_processor = RFM_Processor()
    df_rfm_target = rfm_processor.fit_transform(df_raw)

    # Add RFM features to numerical list
    final_numerical_features = NUMERICAL_FEATURES + RFM_FEATURES

    # Data for WoE transformer (contains all features + target)
    X_woe = df_rfm_target[[col for col in df_rfm_target.columns if col not in [TARGET_COLUMN, CUSTOMER_ID]]].copy()
    y_woe = df_rfm_target[TARGET_COLUMN]

    # 2. WoE Transformation (Task 3)
    woe_transformer = WoETransformer(categorical_features=CATEGORICAL_FEATURES)
    woe_transformer.fit(X_woe, y_woe)
    X_transformed = woe_transformer.transform(X_woe)

    # 3. Standardization (Task 3)
    scaler = StandardScaler()
    X_transformed[final_numerical_features] = scaler.fit_transform(X_transformed[final_numerical_features])

    # Final dataset
    final_data = X_transformed.copy()
    final_data[TARGET_COLUMN] = y_woe.values

    return final_data, rfm_processor, woe_transformer, scaler


if __name__ == '__main__':
    # This is for testing the script locally
    # MAKE SURE YOUR DATA FILE PATH IS CORRECT
    DATA_PATH = '../data/raw/transactions.csv'

    processed_df, rfm_p, woe_t, scaler = get_processed_data(DATA_PATH)
    print("\nProcessed Data Head:")
    print(processed_df.head())
    print("\nTarget Distribution:")
    print(processed_df[TARGET_COLUMN].value_counts(normalize=True))

    # You would typically save the processed data and the transformers here
    # processed_df.to_csv('../data/processed/model_data.csv', index=False)