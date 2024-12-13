"""
This Streamlit application uses machine learning models for property prediction after preprocessing the data.

The program consists of the following steps:
1. **Preprocessing**: The `Cleaning` class is imported from the `cleaning_data` module to clean and preprocess the data. It returns three items:
   - `reverse_mappings`: A dictionary of reverse mappings for categorical data.
   - `reference_data`: A reference dataset used for prediction.
   - `mappings`: A dictionary containing mappings for encoding categorical data.
   
2. **Prediction**: The `Prediction` class is imported from the `prediction` module. It uses the data provided by the preprocessing step to make predictions.
   - `Program.predict()`: Makes predictions based on the processed data (reverse mappings, reference data, and mappings).

Dependencies:
- `pandas`, `numpy`: For data handling and numerical operations.
- `joblib`: For loading pre-trained models.
- `sklearn.preprocessing.MinMaxScaler`: For scaling features.
- `sys` and `os`: For managing system paths and accessing files.

File paths are set dynamically using `sys.path.append` to ensure the correct modules from the `preprocessing` and `predict` directories are imported.

Example usage:
1. The `Cleaning` class is instantiated to preprocess the data.
2. The `Prediction` class is instantiated to generate predictions from the cleaned data.

Modules:
- `cleaning_data` (preprocessing module): Contains the `Cleaning` class for data cleaning and preprocessing.
- `prediction` (prediction module): Contains the `Prediction` class for making property predictions.

How to Run:
1. Save this script as `app.py` (or your preferred filename).
2. Open your terminal and navigate to the directory where `app.py` is located.
3. Run the application using Streamlit by typing the following command in the terminal:
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.append('/home/learner/Desktop/D 13 Dec/Immo_app/Predict')
sys.path.append('/home/learner/Desktop/D 13 Dec/Immo_app/preprocessing')

# Get the absolute path to the preprocessing folder
preprocessing_path = os.path.join(
    os.path.dirname(__file__),  # Current script directory
    'preprocessing'  # Preprocessing folder name
)

from cleaning_data import Cleaning
from prediction import Prediction


c= Cleaning()
Program = Prediction()
reverse_mappings,reference_data,mappings=c.preprocess()
Program.predict(reverse_mappings,reference_data,mappings)