# This script demonstrates a complete machine learning pipeline for fraud detection,
# tailored for an ML Engineer role assessment. It covers data loading, cleaning,
# feature engineering, model training, and evaluation, and now includes a
# simple, production-ready REST API for predictions.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, jsonify
import numpy as np
import os
import joblib

# Define paths for saving artifacts (model and encoder)
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'one_hot_encoder.joblib')


# --- 1. DATA LOADING & INITIAL SETUP ---
def load_data(file_path):
    """Loads the raw e-commerce data from a CSV file."""
    print("Step 1: Loading data...")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return None


# --- 2. DATA CLEANING & PREPROCESSING ---
def preprocess_data(df):
    """
    Performs data cleaning and initial preprocessing steps.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print("\nStep 2: Preprocessing and cleaning data...")

    # 2.1 Drop rows where the target variable 'suspected_fraud' is missing.
    initial_rows = len(df)
    df.dropna(subset=['suspected_fraud'], inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with missing 'suspected_fraud' values.")

    # 2.2 Standardize the 'country_code' column.
    # Replace common variations of UK with GB (standard) and fill missing as UNKNOWN
    df['country_code'] = df['country_code'].str.upper().str.strip()
    df['country_code'].replace({'UK': 'GB', 'ENGLAND': 'GB', 'SCOTLAND': 'GB', 'WALES': 'GB'}, inplace=True)
    df['country_code'].fillna('UNKNOWN', inplace=True)
    print("Standardized and filled missing 'country_code' values.")

    # 2.3 Convert 'order_date' to datetime objects.
    df['order_date'] = pd.to_datetime(df['order_date'])
    print("Converted 'order_date' to datetime format.")

    # 2.4 Convert 'suspected_fraud' to a numerical target variable (0 and 1).
    df['suspected_fraud'] = df['suspected_fraud'].map({'no': 0, 'yes': 1})
    print("Converted 'suspected_fraud' to numerical target (0=not fraud, 1=fraud).")

    return df


# --- 3. FEATURE ENGINEERING ---
def engineer_features(df):
    """
    Creates new features from existing columns.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new engineered features.
    """
    print("\nStep 3: Engineering new features...")

    # 3.1 Extract cyclical features from 'order_date'.
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['month'] = df['order_date'].dt.month_name()
    print("Created 'day_of_week' and 'month' features.")

    # 3.2 Extract email domain.
    # Fill any missing emails before splitting to avoid error.
    df['email'].fillna('unknown@unknown.com', inplace=True)
    df['email_domain'] = df['email'].str.split('@').str[1]
    print("Extracted 'email_domain' feature.")

    return df


# --- 4. MODEL TRAINING AND SAVING ---
def train_and_save_model(df, numerical_features, categorical_features, target):
    """
    Trains a model, evaluates it, and saves the model and the fitted OneHotEncoder.
    This ensures that the deployed model uses the exact same transformation rules.
    """
    print("\nStep 4: Training and saving model and encoder...")

    # Initialize OneHotEncoder and fit on training data
    # We set handle_unknown='ignore' to prevent errors if a new, unseen category appears in prediction data.
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 1. Prepare data for encoding
    X_cat = df[categorical_features]
    y = df[target]

    # 2. Fit encoder and transform data
    ohe.fit(X_cat)
    encoded_features = ohe.transform(X_cat)
    encoded_feature_names = ohe.get_feature_names_out(categorical_features)

    # 3. Create the final feature set
    X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # NEW FIX: Reset index for clean concatenation in both training and prediction
    X_encoded.reset_index(drop=True, inplace=True)
    X_numerical = df[numerical_features].reset_index(drop=True)

    X = pd.concat([X_encoded, X_numerical], axis=1)


    # Explicitly ensure the original categorical features are NOT in the final feature list X
    cols_to_drop_if_present = [col for col in categorical_features if col in X.columns]
    if cols_to_drop_if_present:
        X.drop(columns=cols_to_drop_if_present, inplace=True)
        print(f"DEBUG: Dropped stray base categorical columns: {cols_to_drop_if_present} from feature set X.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully.")

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"--- Model Evaluation (on the test set) ---")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    # --- Save Artifacts (Model and Encoder) ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(ohe, ENCODER_PATH)
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'feature_names.joblib'))  # Save feature list for alignment

    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")


# --- 5. FLASK API WRAPPER ---
app = Flask(__name__)

# Global variables to hold the loaded model and encoder
LOADED_MODEL = None
LOADED_OHE = None
LOADED_FEATURES = None


def load_artifacts():
    """Load the model, encoder, and feature names into global variables."""
    global LOADED_MODEL, LOADED_OHE, LOADED_FEATURES

    # Load Model
    if os.path.exists(MODEL_PATH):
        LOADED_MODEL = joblib.load(MODEL_PATH)
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
        return False

    # Load Encoder
    if os.path.exists(ENCODER_PATH):
        LOADED_OHE = joblib.load(ENCODER_PATH)
    else:
        print(f"ERROR: Encoder file not found at {ENCODER_PATH}.")
        return False

    # Load Feature Names list for column alignment
    feature_names_path = os.path.join(MODEL_DIR, 'feature_names.joblib')
    if os.path.exists(feature_names_path):
        LOADED_FEATURES = joblib.load(feature_names_path)
    else:
        print(f"ERROR: Feature names file not found at {feature_names_path}.")
        return False

    print("Model, encoder, and features successfully loaded for API serving.")
    return True


@app.route('/predict_fraud', methods=['POST'])
def predict():
    """
    API endpoint to make a fraud prediction on new order data.
    """
    if LOADED_MODEL is None or LOADED_OHE is None or LOADED_FEATURES is None:
        return jsonify(
            {'error': 'Model artifacts not loaded. Please ensure the training step completed successfully.'}), 503

    data = request.get_json(force=True)

    required_keys = ['order_value', 'country_code', 'order_date', 'email']
    if not data or not all(key in data for key in required_keys):
        return jsonify({'error': f'Invalid input data. Required keys: {", ".join(required_keys)}'}), 400

    try:
        # 1. Create a DataFrame from the incoming JSON data
        df = pd.DataFrame([data])

        # NEW FIX: Reset index for clean concatenation in prediction
        df.reset_index(drop=True, inplace=True)

        # 2. Apply the same feature engineering and cleaning as training
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['country_code'] = df['country_code'].str.upper().str.strip()
        df['country_code'].replace({'UK': 'GB', 'ENGLAND': 'GB', 'SCOTLAND': 'GB', 'WALES': 'GB'}, inplace=True)
        df['country_code'].fillna('UNKNOWN', inplace=True)

        df['day_of_week'] = df['order_date'].dt.day_name()
        df['month'] = df['order_date'].dt.month_name()

        df['email'].fillna('unknown@unknown.com', inplace=True)
        df['email_domain'] = df['email'].str.split('@').str[1]

        # 3. Identify feature sets
        numerical_features = ['order_value']
        categorical_features = ['country_code', 'day_of_week', 'month', 'email_domain']

        # 4. Transform categorical features using the LOADED ENCODER
        X_cat = df[categorical_features]
        encoded_features = LOADED_OHE.transform(X_cat)
        encoded_feature_names = LOADED_OHE.get_feature_names_out(categorical_features)

        X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

        # 5. Combine and align features (CRITICAL STEP)
        X_encoded.reset_index(drop=True, inplace=True)
        X_numerical = df[numerical_features].reset_index(drop=True)

        X = pd.concat([X_encoded, X_numerical], axis=1)

        # Align columns to match the training data using the loaded feature list
        X_final = X.reindex(columns=LOADED_FEATURES, fill_value=0)

        # 6. Make a prediction
        prediction = LOADED_MODEL.predict(X_final)[0]
        prediction_text = "fraudulent" if prediction == 1 else "not fraudulent"

        return jsonify({'prediction': prediction_text}), 200

    except Exception as e:
        return jsonify({'error': f'Internal Server Error during prediction: {str(e)}'}), 500


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    DATASET_PATH = 'dirty_uk_ecommerce_orders_v3.csv'

    # 1. Check/Train Model
    if not os.path.exists(MODEL_PATH):
        df = load_data(DATASET_PATH)
        if df is not None:
            df = preprocess_data(df)
            df = engineer_features(df)

            NUMERICAL_FEATURES = ['order_value']
            CATEGORICAL_FEATURES = ['country_code', 'day_of_week', 'month', 'email_domain']
            TARGET = 'suspected_fraud'

            train_and_save_model(df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET)

    # 2. Load Artifacts and Run API
    if load_artifacts():
        print("\nStarting Flask API...")
        print("Send a POST request to http://127.0.0.1:5000/predict_fraud")
        print(
            "Example JSON body: {'order_value': 1500.0, 'country_code': 'GB', 'order_date': '2025-05-15', 'email': 'test@example.com'}")

        # NOTE: use_reloader=False is crucial in this environment to prevent errors
        # Flask is also set to run in debug mode for better error visibility
        app.run(debug=True, use_reloader=False)
