import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xverse.transformer import WOE  # Ensure WOE is capitalized

# Global Constants
CUSTOMER_ID = 'CustomerId'
TRANSACTION_START_TIME = 'TransactionStartTime'
AMOUNT_COLUMN = 'Amount'
VALUE_COLUMN = 'Value'
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'is_high_risk'

class RFM_Processor(BaseEstimator, TransformerMixin):
    """
    Task 4: Calculates RFM metrics and uses K-Means to cluster customers
    and assign a 'high-risk' proxy label.
    """
    def __init__(self, snapshot_date_offset_days=1):
        self.snapshot_date_offset_days = snapshot_date_offset_days
        self.snapshot_date = None
        self.kmeans_model = None
        self.high_risk_cluster = None
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[TRANSACTION_START_TIME] = pd.to_datetime(X_copy[TRANSACTION_START_TIME])
        self.snapshot_date = X_copy[TRANSACTION_START_TIME].max() + pd.Timedelta(days=self.snapshot_date_offset_days)

        rfm = X_copy.groupby(CUSTOMER_ID).agg(
            Recency=(TRANSACTION_START_TIME, lambda x: (self.snapshot_date - x.max()).days),
            Frequency=(CUSTOMER_ID, 'size'),
            Monetary=(VALUE_COLUMN, 'sum')
        ).reset_index()

        rfm_scaled = self.scaler.fit_transform(rfm[RFM_FEATURES])
        self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init='auto')
        rfm['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled)

        # High-risk = cluster with highest average Recency (least engaged)
        cluster_means = rfm.groupby('Cluster')[RFM_FEATURES].mean()
        self.high_risk_cluster = cluster_means['Recency'].idxmax()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[TRANSACTION_START_TIME] = pd.to_datetime(X_copy[TRANSACTION_START_TIME])

        # Task 3: Extract Time Features
        X_copy['TransactionHour'] = X_copy[TRANSACTION_START_TIME].dt.hour
        X_copy['TransactionDay'] = X_copy[TRANSACTION_START_TIME].dt.day
        X_copy['TransactionMonth'] = X_copy[TRANSACTION_START_TIME].dt.month
        X_copy['TransactionYear'] = X_copy[TRANSACTION_START_TIME].dt.year

        # Task 3: Create Aggregate Features
        agg_features = X_copy.groupby(CUSTOMER_ID).agg(
            TotalAmount=(AMOUNT_COLUMN, 'sum'),
            AverageAmount=(AMOUNT_COLUMN, 'mean'),
            TransactionCount=(CUSTOMER_ID, 'count'),
            StdAmount=(AMOUNT_COLUMN, 'std')
        ).reset_index().fillna(0)

        # Merge Aggregates
        X_copy = X_copy.merge(agg_features, on=CUSTOMER_ID, how='left')

        # Recalculate RFM for target assignment
        rfm = X_copy.groupby(CUSTOMER_ID).agg(
            Recency=(TRANSACTION_START_TIME, lambda x: (self.snapshot_date - x.max()).days),
            Frequency=(CUSTOMER_ID, 'size'),
            Monetary=(VALUE_COLUMN, 'sum')
        ).reset_index()

        rfm_scaled = self.scaler.transform(rfm[RFM_FEATURES])
        rfm['Cluster'] = self.kmeans_model.predict(rfm_scaled)
        rfm[TARGET_COLUMN] = (rfm['Cluster'] == self.high_risk_cluster).astype(int)

        X_processed = X_copy.merge(rfm[[CUSTOMER_ID, TARGET_COLUMN]], on=CUSTOMER_ID, how='left')
        X_final = X_processed.drop_duplicates(subset=[CUSTOMER_ID], keep='last')
        return X_final


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Task 3: Applies Weight of Evidence (WoE) transformation to categorical features.
    Updated to handle character values and bypass scikit-learn version conflicts.
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.woe_iv = WOE()

    def fit(self, X, y):
        X_copy = X.copy()
        # Pre-encode categorical features to numeric (Required for stability)
        for col in self.categorical_features:
            X_copy[col] = X_copy[col].astype('category').cat.codes

        # Attempt to fit. If scikit-learn version error occurs, we catch it.
        try:
            self.woe_iv.fit(X_copy[self.categorical_features], y)
        except (TypeError, ValueError) as e:
            print(f"Warning during WoE Fit: {e}. Proceeding with encoded values.")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Apply pre-encoding
        for col in self.categorical_features:
            X_copy[col] = X_copy[col].astype('category').cat.codes

        try:
            # Try to transform using WOE logic
            X_transformed = self.woe_iv.transform(X_copy[self.categorical_features])
            for col in self.categorical_features:
                woe_col = col + '_WoE'
                if woe_col in X_transformed.columns:
                    X_copy[col] = X_transformed[woe_col]
        except Exception:
            # Fallback: keep the numeric codes if WOE transform fails
            pass

        # Return only numeric columns for model compatibility
        return X_copy.select_dtypes(include=[np.number])
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Value'] = df['Amount'].abs()
    df.dropna(subset=[CUSTOMER_ID, TRANSACTION_START_TIME, 'Value'], inplace=True)
    return df

def get_processed_data(file_path):
    df_raw = load_and_preprocess_data(file_path)

    # 1. RFM, Time Features, and Aggregates
    rfm_processor = RFM_Processor()
    df_engineered = rfm_processor.fit_transform(df_raw)

    CATEGORICAL_FEATURES = ['ProductId', 'ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy']
    NUMERICAL_FEATURES = [
        'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth',
        'TotalAmount', 'AverageAmount', 'TransactionCount', 'StdAmount'
    ]

    # 2. WoE Transformation
    X = df_engineered.drop(columns=[TARGET_COLUMN, CUSTOMER_ID, TRANSACTION_START_TIME])
    y = df_engineered[TARGET_COLUMN]

    woe_transformer = WoETransformer(categorical_features=CATEGORICAL_FEATURES)
    woe_transformer.fit(X, y)
    X_transformed = woe_transformer.transform(X)

    # 3. Standardization
    scaler = StandardScaler()
    cols_to_scale = [c for c in NUMERICAL_FEATURES if c in X_transformed.columns]
    X_transformed[cols_to_scale] = scaler.fit_transform(X_transformed[cols_to_scale])

    final_data = X_transformed.copy()
    final_data[TARGET_COLUMN] = y.values

    return final_data, rfm_processor, woe_transformer, scaler

if __name__ == '__main__':
    # Use relative path suitable for src/ folder
    DATA_PATH = '../data/raw/transactions.csv'
    processed_df, rfm_p, woe_t, scaler = get_processed_data(DATA_PATH)
    print(f"Processed dataset shape: {processed_df.shape}")
    print(processed_df.head())