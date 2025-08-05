import pandas as pd

def preprocess_data(file_path) -> pd.DataFrame:
    """
    Preprocess the dataset by loading it, handling missing values, and encoding categorical variables.
    
    Parameters:
    file_path (str): The path to the dataset file.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # Load the dataset
    df = pd.read_csv(file_path)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df