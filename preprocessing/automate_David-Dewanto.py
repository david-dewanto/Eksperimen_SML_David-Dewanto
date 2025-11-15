"""
Automated Preprocessing Script for Iris Dataset
Author: David Dewanto

This script automates the preprocessing pipeline from the experimentation notebook.
It performs:
1. Data loading
2. Missing value handling
3. Duplicate removal
4. Feature engineering
5. Label encoding
6. Feature scaling
7. Train-test split

Returns: Preprocessed data ready for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load the raw dataset

    Args:
        file_path (str): Path to the raw CSV file

    Returns:
        pd.DataFrame: Raw dataset
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def handle_missing_values(df, numerical_features):
    """
    Handle missing values in the dataset

    Args:
        df (pd.DataFrame): Input dataframe
        numerical_features (list): List of numerical feature names

    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("\nHandling missing values...")
    missing_before = df.isnull().sum().sum()

    if missing_before > 0:
        for col in numerical_features:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        print(f"Handled {missing_before} missing values")
    else:
        print("No missing values found")

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    print("\nRemoving duplicates...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    removed = before - after
    print(f"Removed {removed} duplicate rows")
    return df


def feature_engineering(df):
    """
    Create new features from existing ones

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with new features
    """
    print("\nPerforming feature engineering...")

    # Create area features
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']

    # Create ratio features
    df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']

    print("Created 4 new features:")
    print("  - sepal_area")
    print("  - petal_area")
    print("  - sepal_ratio")
    print("  - petal_ratio")

    return df


def encode_target(df, target_column='species'):
    """
    Encode the target variable

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column

    Returns:
        tuple: (df with encoded target, label_encoder object)
    """
    print(f"\nEncoding target variable: {target_column}")

    label_encoder = LabelEncoder()
    df['target_encoded'] = label_encoder.fit_transform(df[target_column])

    print("Label encoding mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {idx}")

    return df, label_encoder


def scale_features(df, feature_columns):
    """
    Scale features using StandardScaler

    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of columns to scale

    Returns:
        tuple: (scaled dataframe, scaler object)
    """
    print("\nScaling features...")

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

    print(f"Scaled {len(feature_columns)} features using StandardScaler")

    return df_scaled, scaler


def split_data(df, feature_columns, target_column='target_encoded',
               test_size=0.2, random_state=42):
    """
    Split data into training and testing sets

    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of feature column names
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(input_file, output_file=None, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline

    Args:
        input_file (str): Path to raw data file
        output_file (str): Path to save preprocessed data (optional)
        test_size (float): Proportion of test set
        random_state (int): Random seed

    Returns:
        dict: Dictionary containing:
            - X_train, X_test, y_train, y_test
            - df_preprocessed (full preprocessed dataframe)
            - scaler, label_encoder
            - feature_columns
    """
    print("="*80)
    print("STARTING AUTOMATED PREPROCESSING PIPELINE")
    print("="*80)

    # Define feature columns
    numerical_features = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]

    # Step 1: Load data
    df = load_data(input_file)

    # Step 2: Handle missing values
    df = handle_missing_values(df, numerical_features)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Feature engineering
    df = feature_engineering(df)

    # Step 5: Encode target
    df, label_encoder = encode_target(df, 'species')

    # Define all feature columns (original + engineered)
    feature_columns = [
        'sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)',
        'sepal_area', 'petal_area',
        'sepal_ratio', 'petal_ratio'
    ]

    # Step 6: Scale features
    df_scaled, scaler = scale_features(df, feature_columns)

    # Step 7: Split data
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, feature_columns, 'target_encoded', test_size, random_state
    )

    # Save preprocessed data if output file is specified
    if output_file:
        print(f"\nSaving preprocessed data to: {output_file}")
        df_scaled.to_csv(output_file, index=False)
        print("Data saved successfully")

    # Print summary
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED")
    print("="*80)
    print(f"Total samples: {df_scaled.shape[0]}")
    print(f"Total features: {len(feature_columns)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Target classes: {label_encoder.classes_.tolist()}")
    print("="*80)

    # Return all components
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'df_preprocessed': df_scaled,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'numerical_features': numerical_features
    }


if __name__ == "__main__":
    # Example usage
    input_path = '../iris_raw.csv'
    output_path = 'iris_preprocessing.csv'

    # Run preprocessing pipeline
    results = preprocess_pipeline(
        input_file=input_path,
        output_file=output_path,
        test_size=0.2,
        random_state=42
    )

    print("\nPreprocessing completed successfully!")
    print(f"Access the results using the returned dictionary:")
    print(f"  - results['X_train']: Training features")
    print(f"  - results['X_test']: Testing features")
    print(f"  - results['y_train']: Training labels")
    print(f"  - results['y_test']: Testing labels")
    print(f"  - results['df_preprocessed']: Full preprocessed dataframe")
    print(f"  - results['scaler']: Fitted StandardScaler")
    print(f"  - results['label_encoder']: Fitted LabelEncoder")
