import requests
import pandas as pd
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

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


def nan_series_handling(series: pd.Series, data_type: str='Int64') -> pd.Series:
    # Step 1: Convert to numeric, coercing errors to NaN
    series = pd.to_numeric(series, errors='coerce')

    # Step 2: Convert to integer, which will turn NaN to None
    series = series.astype(data_type)

    # Step 3: Replace any remaining NaN or pd.NA with None
    series = series.replace({pd.NA: None})

    return series


def setup_logging(model_name, root_dir):
    # set up logging for the current process
    process_id = os.getpid()
    # log_file = f"wfcv_log_{process_id}_{model_name.replace(' ', '')}"
    log_file = f"wfcv_log_{model_name.replace(' ', '')}.txt"
    log_path = os.path.join(root_dir, log_file)
    
    # configure logging to write to a specific file
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s'
    )

def analyze_sentiment_finbert(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_prob = probabilities[0][2].item()
    negative_prob = probabilities[0][0].item()
    
    return positive_prob - negative_prob  # Returns a score between -1 and 1