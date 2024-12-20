import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_val_split(dataset, embedding_column, test_size=0.1, val_size=0.1, random_state=42):
    """
    Split the dataset into training, testing, and validation sets in an 8:1:1 ratio.
    """
    # Convert the embeddings into separate columns
    embeddings_df = pd.concat([dataset[embedding_column].apply(pd.Series)], axis=1)
    dataset = pd.concat([dataset.drop(embedding_column, axis=1), embeddings_df], axis=1)

    # Split into X and y
    X = dataset.drop('Target', axis=1)  # all columns except the 'Target' column
    y = dataset['Target']              # the 'Target' column

    print(f"Shape of feature vectors (X): {X.shape}")
    print(f"Shape of target labels (y): {y.shape}")
    
    # Split into 8:1:1 ratio
    X_train, X_test_temp, y_train, y_test_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y
    )
    
    val_ratio_adjusted = val_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_temp, y_test_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_test_temp
    )
    
    return X_train, X_test, X_val, y_train, y_test, y_val