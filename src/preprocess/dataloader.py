"""
dataloader.py

A utility for loading and splitting the Telco Customer Churn dataset using pandas DataFrames.
"""
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

def load_telco_data(
    path: str = "data/data.csv",
    test_size: float = 0.2,
    val_size: Optional[float] = 0.1,
    random_state: int = 42,
    stratify_col: Optional[str] = "Churn"
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Loads the Telco Customer Churn dataset and splits it into train, validation, and test sets.
    Returns (train_df, test_df, val_df) if val_size is provided, else (train_df, test_df, None).
    """
    df = pd.read_csv(path)
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    val_df = None
    if val_size:
        # Split train into train/val
        stratify_train = train_df[stratify_col] if stratify_col and stratify_col in train_df.columns else None
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=stratify_train)
    return train_df, test_df, val_df

if __name__ == "__main__":
    train, test, val = load_telco_data()
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape if val is not None else 'N/A'}")
    print(f"Test shape: {test.shape}")
