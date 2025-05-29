import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded with shape: {df.shape}")
    return df


def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Removed NaN and duplicate rows. New shape: {df.shape}")
    return df


def handle_outliers(df):
    df = df[
        (df['person_age'] <= 70)
        & (df['person_income'] <= 3_000_000)
        & (df['person_emp_exp'] <= 40)
    ]
    print(f"Outliers removed. New shape: {df.shape}")
    return df


def bin_credit_score(score):
    if 800 <= score <= 850:
        return 'Exceptional'
    elif 740 <= score <= 799:
        return 'Very Good'
    elif 670 <= score <= 739:
        return 'Good'
    elif 580 <= score <= 669:
        return 'Fair'
    return 'Poor'


def bin_age(age):
    if age < 18:
        return 'Under 18'
    elif 18 <= age <= 29:
        return '18-29'
    elif 30 <= age <= 49:
        return '30-49'
    return '50+'


def apply_binning(df):
    df['credit_score_binning'] = df['credit_score'].apply(bin_credit_score)
    df['person_age_binning'] = df['person_age'].apply(bin_age)
    df.dropna(inplace=True)
    print(f"Binning applied. Current shape: {df.shape}")
    return df


def encode_categorical(df):
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])
    print("Categorical encoding completed.")
    return df


def standardize_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print("Standardization complete.")
    return df


def split_data(df, target_col='loan_status'):
    X = df.drop(columns=['person_age', 'credit_score', target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def save_to_csv(X_train, X_test, y_train, y_test):
    pd.concat([X_train, y_train], axis=1).to_csv('preprocessing/train_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('preprocessing/test_data.csv', index=False)
    print("Data saved to train_data.csv and test_data.csv")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    df_raw = load_data('loan_data.csv')
    df_cleaned = clean_data(df_raw)
    df_no_outliers = handle_outliers(df_cleaned)
    df_binned = apply_binning(df_no_outliers)
    df_encoded = encode_categorical(df_binned)

    standardize_cols = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file',
        'credit_score_binning', 'person_age_binning'
    ]

    df_final = standardize_features(df_encoded, standardize_cols)
    X_train, X_test, y_train, y_test = split_data(df_final)
    save_to_csv(X_train, X_test, y_train, y_test)