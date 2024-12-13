
"""
This script is a Streamlit application for predicting property prices using a pre-trained model.

Features:
- Accepts user input for various property attributes.
- Normalizes the input data based on reference data.
- Uses a pre-trained model to predict property prices.
- Displays the predicted price in the Streamlit app.

Modules:
- pandas: For data manipulation.
- numpy: For numerical operations.
- joblib: For loading the pre-trained model.
- pathlib: For handling file paths.
- cleaning_data: For preprocessing and normalization.

Classes:
- Prediction: Handles property price prediction workflow.

Functions:
- preprocess: Initializes preprocessing by cleaning and encoding data.
- normalize_data: Normalizes the input data to match the reference data.

How to Use:
1. Run this script using `streamlit run <script_name>.py`.
2. Enter property features in the input fields.
3. Click on "Predict" to get the estimated property price.
"""


import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import sys
import os



sys.path.append('/workspaces/Immo_app/preprocessing')
# Get the absolute path to the preprocessing folder
preprocessing_path = os.path.join(
    os.path.dirname(__file__),  # Current script directory
    'preprocessing'  # Preprocessing folder name
)

from preprocessing.cleaning_data import Cleaning

class Prediction :
    """
        Initializes the Prediction class.
        """
    def __init__(self):
        pass

    def predict(self,reverse_mappings,reference_data,mappings):
        
        """
        Collects user input for property features, preprocesses the data, and predicts the property price.

        Args:
            reverse_mappings (dict): A dictionary mapping encoded categorical values back to their original form.
            reference_data (pd.DataFrame): A reference DataFrame used for normalizing input data.
            mappings (dict): A dictionary mapping categorical features to their encoded values.

        Returns:
            None: Displays the predicted price directly in the Streamlit app.
        """

        
        
        # Manual input for single prediction
        st.title("House Price Prediction App")
        st.write("""
        This application predicts house prices using a pre-trained CatBoost model. 
        Enter the required features to get a prediction.
        """)

        st.subheader("Enter Features for Prediction")
        manual_input = {}
        model = load("/workspaces/Immo_app/model/model_Hussain.joblib")
        c=Cleaning()
        # Handle input for features
        for column in reference_data.columns:
            if column in mappings:  # Categorical column
                options = list(mappings[column].values())
                user_input = st.selectbox(f"Select {column.split('_encoded')[0]} of the property:", options)
                manual_input[column] = reverse_mappings[column][user_input]
            elif column == "Bedrooms":  # Integer input for Bedrooms
                manual_input[column] = st.number_input(f"Enter {column} of the property (integer):", min_value=0, value=1)
            elif column == "Living_Area":  # Integer input for Living_Area (cannot be zero)
                manual_input[column] = st.number_input(f"Enter the living area in squared meter (integer, cannot be zero):", min_value=1, value=50)
            elif column == "Facades":  # Integer input for Facades
                manual_input[column] = st.number_input(f"Enter {column} (integer):", min_value=1, value=1)
            elif column == "Is_Equiped_Kitchen":  # Boolean input
                manual_input[column] = st.checkbox("Does it have an equipped kitchen?")

            elif column == "Terrace":  # Boolean input
                manual_input[column] = st.checkbox("Does it have a terrace?")

            elif column == "Garden":  # Boolean input
                manual_input[column] = st.checkbox("Does it have a garden?")

            elif column in ["GDP", "Type_encoded","Avg_rent", "Avg price","Bedrooms_per_area","Is_On_Coast","Prov_encoded","Region_encoded"]:
                manual_input[column] = 0
                continue

            else:  # Numerical columns
                manual_input[column] = st.number_input(f"Enter {column}:", value=0.0)

        # Calculate Bedrooms per Area (Living_Area / Bedrooms), ensuring Bedrooms is greater than zero
        if manual_input["Bedrooms"] > 0:
            manual_input["Bedrooms_per_area"] = manual_input["Living_Area"] / manual_input["Bedrooms"]
        else:
            manual_input["Bedrooms_per_area"] = 0


        # Prediction button for manual input
        if st.button("Predict"):
            # Create a DataFrame from manual input
            manual_data = pd.DataFrame([manual_input])
            # Normalize the input data (excluding 'Price' and 'Id' columns)
            normalized_manual_data = c.normalize_data(manual_data, reference_data)
            # Predict using the model
            prediction = np.expm1(model.predict(normalized_manual_data)[0])

            # Display the prediction
            st.write(f"Predicted Price: €{int(prediction):,}")



#reverse_mappings,reference_data,mappings=preprocess()

#predict(reverse_mappings,reference_data,mappings)