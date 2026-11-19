
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor


# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Convert actual log prices back to rupees
actual_prices = np.expm1(y_test)


# ============================================================
# 2. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
}


# ============================================================
# 3. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

os.makedirs("data/models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}...")
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predict on log scale
    predictions_log = model.predict(X_test)

    # Convert predictions back to rupees
    predictions = np.expm1(predictions_log)

    # Prevent negative price predictions
    predictions = np.maximum(predictions, 0)

    # --------------------------------------------------------
    # Original price scale evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_prices,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(actual_prices, predictions)
    )

    r2 = r2_score(
        actual_prices,
        predictions
    )

    # --------------------------------------------------------
    # Log scale evaluation
    # --------------------------------------------------------

    log_mae = mean_absolute_error(
        y_test,
        predictions_log
    )

    log_rmse = np.sqrt(
        mean_squared_error(y_test, predictions_log)
    )

    log_r2 = r2_score(
        y_test,
        predictions_log
    )

    # Store results
    results.append({

        "Model": name,

        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,

        "Log MAE": log_mae,
        "Log RMSE": log_rmse,
        "Log R2 Score": log_r2

    })

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    model_path = os.path.join(
        "data/models",
        filename
    )

    joblib.dump(model, model_path)

    # Display results
    print(f"\n{name} completed!")

    print("\nOriginal Price Scale:")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    print("\nLog Price Scale:")
    print(f"Log MAE: {log_mae:.4f}")
    print(f"Log RMSE: {log_rmse:.4f}")
    print(f"Log R2 Score: {log_r2:.4f}")

    print(f"\nModel saved: {model_path}")


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Sort by original-scale MAE
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))


# Save comparison report
comparison_path = "data/processed/model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 5. IDENTIFY BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model["Model"])
print(f"MAE: ₹{best_model['MAE']:,.2f}")
print(f"RMSE: ₹{best_model['RMSE']:,.2f}")
print(f"R2 Score: {best_model['R2 Score']:.4f}")
print(f"Log R2 Score: {best_model['Log R2 Score']:.4f}")


# ============================================================
# 6. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved model files:")
print("- linear_regression.pkl")
print("- random_forest.pkl")
print("- xgboost.pkl")
print("- preprocessor.pkl")

print("\nComparison report:")
print(comparison_path)

print("\nBest performing model:", best_model["Model"])
