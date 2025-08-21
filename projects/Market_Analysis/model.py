import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_csv("historical_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    df[col] = df[col].str.replace(",", "").astype(float)

# Feature engineering
df["Return"] = (df["Close"] - df["Open"]) / df["Open"]
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df["Volatility"] = df["Close"].rolling(window=5).std()

# Targets
df["Target_reg"] = df["Close"].shift(-1)  # Regression: next day's close
df["Target_clf"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # Classification: up/down

# Drop rows with NaN (from rolling and shifting)
df.dropna(inplace=True)

# Features for modeling
features = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Return", "MA5", "MA10", "Volatility"]

# Feature scaling
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split data
X = df[features]
y_reg = df["Target_reg"]
y_clf = df["Target_clf"]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, shuffle=False
)

# Regression Model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

# Classification Model
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

# --- Analysis & Visualization ---

# Regression Results
plt.figure(figsize=(10,5))
plt.plot(df.index[-len(y_reg_test):], y_reg_test, label="Actual Close")
plt.plot(df.index[-len(y_reg_test):], y_reg_pred, label="Predicted Close")
plt.title("Regression: Actual vs Predicted Next Day Close Price")
plt.xlabel("Date")
plt.ylabel("Scaled Close Price")
plt.legend()
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"Regression RMSE: {rmse:.4f}")

# Classification Results
acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"Classification Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_clf_test, y_clf_pred))

cm = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = reg_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
plt.figure(figsize=(8,4))
feat_imp.plot(kind="bar")
plt.title("Feature Importance (Regression Model)")
plt.tight_layout()
plt.show()

# Insights
print("\n--- Insights ---")
print(f"Top features influencing next day's close: {feat_imp.index[:3].tolist()}")
print(f"Model predicts next day's price with RMSE of {rmse:.4f} (scaled).")
print(f"Classification model accuracy: {acc:.2%}.")
print("Confusion matrix and classification report show how well the model predicts up/down movement.")
