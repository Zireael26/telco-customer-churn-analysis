"""
Feature Engineering for Telco Customer Churn

This script implements the feature engineering steps validated in the notebook, including encoding, transformations, and new feature creation. It is modular, well-documented, and follows project best practices.
"""
import pandas as pd
import numpy as np
from typing import Optional

def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'customerID' and convert 'TotalCharges' to numeric."""
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables: label encoding for binary, one-hot for multi-class."""
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() == 2:
            df[col] = df[col].astype('category').cat.codes
        else:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features such as tenure groups."""
    df = df.copy()
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=['0-12', '13-24', '25-48', '49-60', '61+'])
        df = pd.concat([df, pd.get_dummies(df['tenure_group'], prefix='tenure_group')], axis=1)
        df = df.drop('tenure_group', axis=1)
    return df

def advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features, log transformation, and binning."""
    df = df.copy()
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['MonthlyCharges_tenure'] = df['MonthlyCharges'] * df['tenure']
    if 'TotalCharges' in df.columns:
        df['TotalCharges_log'] = np.log1p(df['TotalCharges'].clip(lower=0))
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, np.inf], labels=['Low', 'Medium', 'High'])
        df = pd.concat([df, pd.get_dummies(df['MonthlyCharges_bin'], prefix='MonthlyCharges_bin')], axis=1)
        df = df.drop('MonthlyCharges_bin', axis=1)
    return df

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in sequence."""
    df = preprocess_basic(df)
    df = encode_categoricals(df)
    df = add_features(df)
    df = advanced_features(df)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature engineering for Telco Customer Churn dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    processed_df = feature_engineering_pipeline(df)
    processed_df.to_csv(args.output, index=False)
    print(f"Feature engineering complete. Output saved to {args.output}")
