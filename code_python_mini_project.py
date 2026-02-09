import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import joblib


# Change filename to whatever you uploaded
df = pd.read_csv("Dataset_calory.csv")
df["Gender"] = df["Gender"].map({"Male" : 0 , "Female" : 1})
activity_map = {
    "Sedentary": 1,
    "Lightly Active": 2,
    "Moderately Active": 3,
    "Very Active": 4,
    "Extra Active": 5
}
df["Activity_Level"] = df["Activity_Level"].map(activity_map)
print(df)

#target_col = "y"
target_col = "Activity_Level"
X = df.drop(columns=[target_col])
y = df[target_col].values

print("X shape:", X.shape, "y shape:", y.shape)
print("Feature columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit ONLY on train
X_test_s  = scaler.transform(X_test)       # transform test using train scaler

print("Train:", X_train_s.shape, "Test:", X_test_s.shape)

model = LinearRegression()
model.fit(X_train_s, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # If y_true is constant, ss_tot can be 0; handle safely:
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

y_pred_test = model.predict(X_test_s)

print("Test MSE:", mse(y_test, y_pred_test))
print("Test R2 :", r2_score(y_test, y_pred_test))

joblib.dump(model, "linear_model_mini_project.pkl")
joblib.dump(scaler, "scaler_mini_project.pkl")
print("Saved: linear_model_mini_project.pkl, scaler_mini_project.pkl")


model = joblib.load("linear_model_mini_project.pkl")
scaler = joblib.load("scaler_mini_project.pkl")

y_pred_test_loaded = model.predict(scaler.transform(X_test))
print("Reloaded model Test MSE:", mse(y_test, y_pred_test_loaded))
print("Reloaded model Test R2 :", r2_score(y_test, y_pred_test_loaded))