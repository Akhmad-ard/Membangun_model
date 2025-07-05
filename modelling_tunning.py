import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
path = "./Predict_Student_Performance_preprocessing"

df_train = pd.read_csv(f"{path}/train.csv")
df_test = pd.read_csv(f"{path}/test.csv")

X_train = df_train.drop(columns=['Grades'])
y_train = df_train["Grades"]

X_test = df_test.drop(columns=['Grades'])
y_test = df_test["Grades"]

# Hyperparameter grid (manual tuning)
param_grid = [
  {"n_estimators": 50, "max_depth": 5},
  {"n_estimators": 100, "max_depth": 10},
  {"n_estimators": 150, "max_depth": 15},
]

for params in param_grid:
  with mlflow.start_run():
    model = RandomForestRegressor(
      n_estimators=params["n_estimators"],
      max_depth=params["max_depth"],
      random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Manual Logging
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    print(f"Run with params: {params} => MSE: {mse:.4f}, R2: {r2:.4f}")
