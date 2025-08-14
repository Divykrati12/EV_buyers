# ev_model_train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

# Load Dataset (replace with actual path if needed)
df = pd.read_csv("ev_buyer_data.csv")

# Impute missing values                                              # missing value present
num_cols = ['Age', 'Income']
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Cap outliers
q_hi = df['Income'].quantile(0.99)
df['Income'] = np.where(df['Income'] > q_hi, q_hi, df['Income'])

# Encode driving habits
df['Driving_Habits'] = df['Driving_Habits'].map({'Low': 1, 'Medium': 2, 'High': 3})         # ordinal data

# One-hot encode City
df = pd.get_dummies(df, columns=['City'], drop_first=True)                                  # False

# Define X and y
X = df.drop('Will_Buy_EV', axis=1)                                                          # independent variable
y = df['Will_Buy_EV']                                                                       # target

# Scale numerical columns
scaler = StandardScaler()
X[['Age', 'Income', 'Vehicle_Budget', 'Environmental_Concern']] = scaler.fit_transform(
    X[['Age', 'Income', 'Vehicle_Budget', 'Environmental_Concern']]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# Save best model
with open("ev_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save feature schema
schema = {
    "model": best_name,
    "features": list(X.columns)
}
with open("model_schema.json", "w") as f:
    json.dump(schema, f)
