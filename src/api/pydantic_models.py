from pydantic import BaseModel
from typing import List

class TransactionInput(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: int
    ProductId: str
    ProductCategory: str
    ChannelId: int
    Amount: float
    Value: float
    TransactionStartTime: str # ISO 8601 format string
    PricingStrategy: int
    FraudResult: int

class CustomerData(BaseModel):
    transactions: List[TransactionInput]

class PredictionOutput(BaseModel):
    CustomerId: str
    RiskProbability: float
    CreditScore: int # Placeholder: score from 300 to 850
    RiskCategory: str