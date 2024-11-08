import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import sys
import logging
from dotenv import load_dotenv
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import glob
from sklearn.preprocessing import StandardScaler, RobustScaler

# Save dictionary to a pickle file
def save_pickle_file(my_dict: dict, filepath: str) -> None:
    # with open(filepath, 'wb') as f:
    joblib.dump(my_dict, filepath)

def load_pickle_file(filename: str) -> Dict:
    return joblib.load(filename)

def api_get(url: str) -> Dict[str, Any] | str:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")

def nan_series_handling(series: pd.Series, data_type: str='int') -> pd.Series:
    # Step 1: Convert to numeric, coercing errors to NaN
    series = pd.to_numeric(series, errors='coerce')

    # Step 2: Convert to integer, which will turn NaN to None
    series = series.astype(data_type)

    # Step 3: Replace any remaining NaN or pd.NA with None
    series = series.replace({pd.NA: None})

    return series

def setup_logging(log_file, root_dir, logger_name=None):
    # Create a unique logger name if not provided
    if logger_name is None:
        logger_name = log_file  # Use log file name to uniquely identify the logger

    # Create or retrieve the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Check if the logger has handlers already, if so, clear them to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set up the log file path
    log_path = os.path.join(root_dir, log_file)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Define a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Create a console handler and set the same formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger  # Return the configured logger


def analyze_sentiment_finbert(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_prob = probabilities[0][2].item()
    negative_prob = probabilities[0][0].item()
    
    return positive_prob - negative_prob  # Returns a score between -1 and 1

def get_most_recent_folder(directory_path):
    # Get all folders in the directory
    folders = [f for f in glob.glob(os.path.join(directory_path, '*')) if os.path.isdir(f)]
    # Sort folders by creation time (newest first)
    most_recent_folder = max(folders, key=os.path.getctime)
    return most_recent_folder


def drop_corr_pair1(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    df = df.drop(columns=to_drop)

    return df

def winsorize_data(data, lower_percentile_threshold, upper_percentile_threshold):
    """
    Winsorize data by clipping extreme outliers in both directions.
    
    Parameters:
    data (numpy.ndarray or pandas.Series): 1D array
    lower_percentile_threshold (float): Lower percentile threshold for outlier detection
    upper_percentile_threshold (float): Upper percentile threshold for outlier detection
    
    Returns:
    numpy.ndarray or pandas.Series: Winsorized data
    """
    # Winsorize the data using NumPy's nanpercentile()
    winsorized_data = np.clip(data, np.nanpercentile(data, lower_percentile_threshold), np.nanpercentile(data, upper_percentile_threshold))
    return winsorized_data

def standardize_df(data, scaler=RobustScaler()):
    # Separate features and target
    df_index = data.index
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column (target)
    target_name = y.name

    # Standardize the feature columns
    scaler = scaler
    X_scaled = scaler.fit_transform(X)

    # Convert back to a DataFrame and combine with the target column
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled[f'{target_name}'] = y.values  # Add the target column back
    df_scaled.index = df_index

    return df_scaled