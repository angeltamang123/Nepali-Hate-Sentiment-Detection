import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_target_column(df: pd.DataFrame, sentiment_col: str, polarity_col: str, target_col: str = "Target"):
    """
    Creates a new target column by combining sentiment and polarity columns, encodes it,
    and returns the updated DataFrame along with the LabelEncoder.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the sentiment and polarity columns.
    - sentiment_col (str): Name of the column containing sentiment values.
    - polarity_col (str): Name of the column containing polarity values.
    - target_col (str): Name of the new target column to be created (default: "Target").

    Returns:
    - pd.DataFrame: The updated DataFrame with the new target column.
    - LabelEncoder: The fitted LabelEncoder for the target column.
    """
    if sentiment_col not in df.columns or polarity_col not in df.columns:
        raise ValueError(f"Columns {sentiment_col} and {polarity_col} must exist in the DataFrame.")

    # Create the target column by combining sentiment and polarity
    df[target_col] = df[sentiment_col].astype(str) + "_" + df[polarity_col].astype(str)

    # Encode the target column using LabelEncoder
    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])

    # Drop the original sentiment and polarity columns
    df = df.drop(columns=[sentiment_col, polarity_col])

    return df, label_encoder
