"""
Week 1: Load and Clean Data
Load the king county house data CSV and perform data cleaning operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath):
    """
    Load the house data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def clean_data(df):
    """
    Clean the house data.
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Remove outliers
    - Convert data types
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")
    
    # Display info about missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # TODO: Add your cleaning logic here
    # - Handle missing values
    # - Remove outliers
    # - Convert data types
    
    return df


def save_cleaned_data(df, output_path):
    """
    Save cleaned data to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned data
        output_path (str): Path to save the CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    # Define paths
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "kc_house_data.csv"
    output_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    
    # Load data
    df = load_data(raw_data_path)
    
    if df is not None:
        # Clean data
        df_cleaned = clean_data(df)
        
        # Save cleaned data
        save_cleaned_data(df_cleaned, output_path)
