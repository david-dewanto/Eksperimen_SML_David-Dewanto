# Iris Dataset Data Experimentation

Data experimentation project for Iris classification with automated preprocessing pipeline.

## Project Structure

```
Eksperimen_SML_David-Dewanto/
├── iris_raw.csv                              # Raw Iris dataset
├── preprocessing/
│   ├── Eksperimen_David-Dewanto.ipynb       # Experimentation notebook
│   ├── automate_David-Dewanto.py            # Automated preprocessing script
│   └── iris_preprocessing.csv               # Preprocessed dataset
├── .github/workflows/
│   └── preprocessing.yml                     # CI/CD automation workflow
└── requirements.txt                          # Python dependencies
```

## Features

### Data Preprocessing Pipeline
- Missing value handling
- Duplicate removal
- Feature engineering (creates 8 features from original 4)
- Label encoding for target variable
- Feature scaling with StandardScaler
- Train-test split

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

This generates `iris_preprocessing.csv` with engineered features ready for model training.

### Experiment with Jupyter
```bash
jupyter notebook preprocessing/Eksperimen_David-Dewanto.ipynb
```

## Dataset Information

**Input:** Iris dataset with 4 features (sepal length, sepal width, petal length, petal width)

**Output:** Preprocessed dataset with 8 engineered features:
- Original 4 features (scaled)
- 4 additional engineered features

**Target:** 3 iris species (setosa, versicolor, virginica)

## Author

David Dewanto
