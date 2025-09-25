# This script demonstrates a complete machine learning pipeline for fraud detection,
# tailored for an ML Engineer role assessment. It covers data loading, cleaning,
# feature engineering, model training, and evaluation, and now includes a
# simple REST API for predictions.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

# Define a path to save the trained model
MODEL_PATH = 'fraud_detection_model.pkl'


# --- 1. DATA LOADING & INITIAL SETUP ---
def load_data(file_path):
    """
    Loads the raw e-commerce data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
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
    df['country_code'] = df['country_code'].str.upper().str.strip()
    df['country_code'].fillna('UNKNOWN', inplace=True)
    print("Standardized and filled missing 'country_code' values.")

    # 2.3 Convert 'order_date' to datetime objects.
    df['order_date'] = pd.to_datetime(df['order_date'])
    print("Converted 'order_date' to datetime format.")

    # 2.4 Convert 'suspected_fraud' to a numerical target variable.
    df['suspected_fraud'] = df['suspected_fraud'].map({'no': 0, 'yes': 1})
    print("Converted 'suspected_fraud' to numerical values (0 and 1).")

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

    # 3.1 Extract day of the week and month from the 'order_date'.
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['month'] = df['order_date'].dt.month_name()
    print("Created 'day_of_week' and 'month' features.")

    # 3.2 Extract email domain as a feature.
    df['email_domain'] = df['email'].str.split('@').str[1]
    print("Extracted 'email_domain' feature.")

    return df


# --- 4. MODEL TRAINING AND SAVING ---
def train_and_save_model(df, features, target):
    """
    Trains a model, evaluates it, and saves it to a file.

    Args:
        df (pd.DataFrame): The DataFrame containing features and target.
        features (list): A list of feature columns.
        target (str): The name of the target column.
    """
    # Handle categorical features using one-hot encoding
    df_encoded = pd.get_dummies(df[features], columns=['country_code', 'day_of_week', 'month'], drop_first=True)

    X = df_encoded
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nStep 4: Splitting data and training model...")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully.")

    # Make predictions on the test set
    print("\nStep 5: Evaluating model performance...")
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"--- Model Evaluation (on the test set) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save the trained model to a file
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': list(X.columns)}, f)
    print(f"\nModel saved to {MODEL_PATH}")


# --- 5. FLASK API WRAPPER ---
app = Flask(__name__)


@app.route('/predict_fraud', methods=['POST'])
def predict():
    """
    API endpoint to make a fraud prediction on new order data.
    """
    # Load the pre-trained model and feature names
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found. Please run the script to train and save the model first.'}), 404

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    model_features = model_data['features']

    # Get JSON data from the request
    data = request.get_json(force=True)

    if not data or not all(key in data for key in ['order_value', 'country_code', 'order_date']):
        return jsonify({'error': 'Invalid input data. Required keys: order_value, country_code, order_date'}), 400

    try:
        # Create a DataFrame from the incoming JSON data
        df = pd.DataFrame([data])

        # Apply the same preprocessing and feature engineering as the training data
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['country_code'] = df['country_code'].str.upper().str.strip()
        df['day_of_week'] = df['order_date'].dt.day_name()
        df['month'] = df['order_date'].dt.month_name()

        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, columns=['country_code', 'day_of_week', 'month'], drop_first=True)

        # Align the columns to match the training data
        # This is a critical step to prevent errors with missing columns in production
        missing_cols = set(model_features) - set(df_encoded.columns)
        for c in missing_cols:
            df_encoded[c] = 0
        df_encoded = df_encoded[model_features]

        # Make a prediction
        prediction = model.predict(df_encoded)[0]
        prediction_text = "fraudulent" if prediction == 1 else  "not fraudulent"

        return jsonify({'prediction': prediction_text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    DATASET_PATH = 'dirty_uk_ecommerce_orders_v3.csv'

    # Check if the model is already trained and saved.
    # This prevents retraining the model every time the API is run.
    if not os.path.exists(MODEL_PATH):
        df = load_data(DATASET_PATH)
        if df is not None:
            df = preprocess_data(df)
            df = engineer_features(df)
            features_list = ['order_value', 'country_code', 'day_of_week', 'month']
            target_list = 'suspected_fraud'
            train_and_save_model(df, features_list, target_list)

    # Run the Flask app
    print("\nStarting Flask API. Send a POST request to http://127.0.0.1:5000/predict")
    print("Example JSON body: {'order_value': 1500, 'country_code': 'GB', 'order_date': '2025-05-15'}")
    app.run(debug=True, use_reloader=False)
