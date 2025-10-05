"""
Data Preparation Script for Tourism Prediction MLOps Project.
- Loads raw dataset from Hugging Face Hub.
- Cleans and encodes categorical columns.
- Splits into train/test sets with stratification.
- Saves processed data locally and re-uploads to the Hub.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Yash0204/prediction-tourism-mlops/tourism.csv"

print("📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded dataset with shape: {df.shape}")

# Drop unique identifiers
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handle missing values
if df.isnull().sum().any():
    print("⚠️ Missing values found — filling with column medians/modes.")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
print("🔤 Categorical columns encoded.")

# Split into features and target
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✂️ Data split complete: Train={len(Xtrain)}, Test={len(Xtest)}")

# Save locally
os.makedirs("data_splits", exist_ok=True)
Xtrain.to_csv("data_splits/Xtrain.csv", index=False)
Xtest.to_csv("data_splits/Xtest.csv", index=False)
ytrain.to_csv("data_splits/ytrain.csv", index=False)
ytest.to_csv("data_splits/ytest.csv", index=False)

# Upload to Hugging Face
files = [f"data_splits/{f}" for f in os.listdir("data_splits")]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed/{os.path.basename(file_path)}",
        repo_id="Yash0204/prediction-tourism-mlops",
        repo_type="dataset",
    )

print("🎉 All processed files uploaded successfully to Hugging Face!")
