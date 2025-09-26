import unittest
import requests
import json
import subprocess
import time
import pandas as pd
from datetime import datetime
import os
import joblib

# --- Configuration ---
# Assuming the Flask app runs on the default host and port
API_URL = "http://127.0.0.1:5000/predict_fraud"
MODEL_DIR = 'models'

# --- Mock Data for Testing ---
# This data is designed to represent valid input that the API expects.
VALID_INPUT_DATA = {
    "order_value": 850.50,
    "country_code": "gb",
    "order_date": "2025-10-20",
    "email": "test@email.com"
}


# --- Test Functions (from model_train_test_serve.py) ---

def _load_artifacts():
    """Helper to load the saved OneHotEncoder and Feature Names list."""
    encoder_path = os.path.join(MODEL_DIR, 'one_hot_encoder.joblib')
    features_path = os.path.join(MODEL_DIR, 'feature_names.joblib')

    try:
        ohe = joblib.load(encoder_path)
        feature_names = joblib.load(features_path)
        return ohe, feature_names
    except FileNotFoundError as e:
        print(f"Error loading artifact: {e}. Ensure model_train_test_serve.py has been run to train the model.")
        return None, None


def _preprocess_test_data(data, ohe, loaded_features):
    """
    Transforms raw input data (like the API receives) into the format
    expected by the trained model (features, encoding, etc.).
    """
    if ohe is None or loaded_features is None:
        return None

    df = pd.DataFrame([data])

    # 1. Feature Engineering (MUST MATCH TRAINING SCRIPT)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['day_of_week'] = df['order_date'].dt.day_name()
    # FIX: Add 'month' feature engineering
    df['month'] = df['order_date'].dt.month_name()
    df['email_domain'] = df['email'].apply(lambda x: x.split('@')[-1] if isinstance(x, str) else 'unknown')

    # 2. Standardization
    df['country_code'] = df['country_code'].str.upper().replace({'UK': 'GB'})
    df['country_code'].fillna('UNKNOWN', inplace=True)

    # 3. Categorical Encoding (using the loaded encoder)
    # FIX: Include 'month' in the categorical features list
    categorical_features = ['country_code', 'day_of_week', 'month', 'email_domain']

    # Select columns used for encoding
    X_cat = df[categorical_features]

    # Transform the data
    encoded_features = ohe.transform(X_cat)
    encoded_feature_names = ohe.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # 4. Combine numerical features
    X = pd.concat([encoded_df, df[['order_value']].reset_index(drop=True)], axis=1)

    # 5. FINAL CRITICAL STEP: Align columns to match the saved feature list
    X_final = X.reindex(columns=loaded_features, fill_value=0)

    return X_final


# --- Test Case Class ---
class TestFraudDetectionAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Start the Flask server process before running tests."""
        print("\n--- Setting up: Starting Flask Server ---")
        try:
            # We assume the main script is run with 'python model_train_test_serve.py'
            cls.server_process = subprocess.Popen(['python', 'model_train_test_serve.py'])
            # Give the server a moment to start up and load the model
            time.sleep(5)
        except FileNotFoundError:
            # Handle case where python command or script file is not found
            cls.server_process = None
            print(
                "ERROR: Could not start Flask server. Ensure 'model_train_test_serve.py' exists and Python is in PATH.")

        # Load the pre-trained encoder and feature list once
        cls.ohe, cls.loaded_features = _load_artifacts()

    @classmethod
    def tearDownClass(cls):
        """Terminate the Flask server process after running tests."""
        if cls.server_process:
            print("\n--- Tearing down: Stopping Flask Server ---")
            cls.server_process.terminate()
            cls.server_process.wait()

    def test_01_preprocessing_standardization(self):
        """Test that country codes are correctly standardized to 'GB' or 'UNKNOWN'."""
        if self.ohe is None:
            self.skipTest("Artifacts not loaded. Skipping preprocessing tests.")

        test_data = {
            "order_value": 100,
            "country_code": "uk",  # Test lowercase
            "order_date": "2025-01-01",
            "email": "test@domain.com"
        }

        preprocessed_df = _preprocess_test_data(test_data, self.ohe, self.loaded_features)

        # Check that the standardized feature 'country_code_GB' exists and is 1
        self.assertTrue('country_code_GB' in preprocessed_df.columns)
        self.assertEqual(preprocessed_df['country_code_GB'].iloc[0], 1.0, "Should encode 'uk' as 'country_code_GB'")

    def test_02_preprocessing_missing_data(self):
        """Test that missing country data is handled by 'UNKNOWN' category."""
        if self.ohe is None:
            self.skipTest("Artifacts not loaded. Skipping preprocessing tests.")

        test_data = {
            "order_value": 500,
            "country_code": None,  # Missing value
            "order_date": "2025-01-01",
            "email": "test@domain.com"
        }

        preprocessed_df = _preprocess_test_data(test_data, self.ohe, self.loaded_features)

        # Check that the 'country_code_UNKNOWN' category is created and is 1
        self.assertTrue('country_code_UNKNOWN' in preprocessed_df.columns)
        self.assertEqual(preprocessed_df['country_code_UNKNOWN'].iloc[0], 1.0,
                         "Should encode missing country as 'country_code_UNKNOWN'")

    def test_03_api_prediction_success(self):
        """Test the API returns a successful response with the correct format."""
        if self.server_process is None:
            self.skipTest("Server not running. Skipping API tests.")

        try:
            response = requests.post(API_URL, json=VALID_INPUT_DATA)
            self.assertEqual(response.status_code, 200, f"API returned non-200 status: {response.status_code}")

            # Check the JSON structure and key
            response_json = response.json()
            self.assertIn('prediction', response_json)

            # Check the prediction value is one of the valid outcomes
            valid_outcomes = ["fraudulent", "not fraudulent"]
            self.assertIn(response_json['prediction'], valid_outcomes)

        except requests.exceptions.ConnectionError:
            self.fail(f"Could not connect to Flask API at {API_URL}. Is model_train_test_serve.py running?")

    def test_04_api_invalid_input_missing_key(self):
        """Test the API handles a request missing a required key."""
        if self.server_process is None:
            self.skipTest("Server not running. Skipping API tests.")

        # Missing 'order_date' key
        invalid_data = {
            "order_value": 100.0,
            "country_code": "GB"
        }

        response = requests.post(API_URL, json=invalid_data)

        # Check for 400 Bad Request status
        self.assertEqual(response.status_code, 400, "API should return 400 for missing input key")

        # Check for error message
        response_json = response.json()
        self.assertIn('error', response_json)


if __name__ == '__main__':
    # Add email field to the test data for preprocessing tests
    VALID_INPUT_DATA['email'] = 'test@example.com'
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
