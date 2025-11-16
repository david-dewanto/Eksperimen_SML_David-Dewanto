# Transactions Dataset Data Experimentation

Data experimentation project for fraud detection with automated preprocessing pipeline.

## Project Structure

```
Eksperimen_SML_David-Dewanto/
├── transactions.csv                          # Raw Transactions dataset
├── preprocessing/
│   ├── Eksperimen_David-Dewanto.ipynb       # Experimentation notebook
│   ├── automate_David-Dewanto.py            # Automated preprocessing script
│   └── transactions_preprocessing.csv       # Preprocessed dataset
├── .github/workflows/
│   └── preprocessing.yml                     # CI/CD automation workflow
└── requirements.txt                          # Python dependencies
```

## Features

### Data Preprocessing Pipeline
- Missing value handling
- Duplicate removal
- Exploratory Data Analysis (EDA)
- Feature engineering (creates additional features from transaction data)
- Feature selection (13 numerical features)
- Label encoding for target variable (fraud detection)
- Data validation and quality checks

### Automation
- GitHub Actions workflow for automated preprocessing
- Runs on push to main branch
- Generates and uploads preprocessed data as artifacts

## Setup

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage

### Run Preprocessing Locally
```bash
cd preprocessing
python automate_David-Dewanto.py
```

This generates `transactions_preprocessing.csv` with engineered features ready for model training.

### Experiment with Jupyter
```bash
jupyter notebook preprocessing/Eksperimen_David-Dewanto.ipynb
```

## Dataset Information

**Input:** Transactions dataset with transaction and user information

**Output:** Preprocessed dataset with 13 numerical features:
- account_age_days
- total_transactions_user
- avg_amount_user
- amount
- promo_used
- avs_match
- cvv_result
- three_ds_flag
- shipping_distance_km
- amount_transactions_product
- amount_avg_product
- amount_avg_ratio
- shipping_age_ratio

**Target:** Binary classification (0 = non-fraud, 1 = fraud)

**Task:** Fraud detection in financial transactions

## Author

David Dewanto
