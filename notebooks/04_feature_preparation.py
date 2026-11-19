
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_house_prices.csv")

print("Original dataset shape:", df.shape)

# 2. Remove identifier and low-value columns
df = df.drop(columns=["source_listing_id", "status"], errors="ignore")

# 3. Separate target and input features
X = df.drop(columns=["price_rupees"])
y = np.log1p(df["price_rupees"])

# 4. Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("\nCategorical features:", categorical_features)

# 5. Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 8. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# 9. Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 10. Save prepared data and preprocessing pipeline
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

from scipy.sparse import issparse

if issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

np.save("data/processed/X_train.npy", X_train_processed)
np.save("data/processed/X_test.npy", X_test_processed)
np.save("data/processed/y_train.npy", y_train.to_numpy())
np.save("data/processed/y_test.npy", y_test.to_numpy())

import joblib
joblib.dump(preprocessor, "data/models/preprocessor.pkl")

print("\n========== PREPROCESSING COMPLETED ==========")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Training and testing datasets saved successfully!")
