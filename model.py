import sys
import joblib

# Friendly import checks
try:
    import pandas as pd
except Exception as e:
    print("Error: required package 'pandas' is not installed.")
    print("Run: python -m pip install pandas")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    print("Error: required package 'scikit-learn' is not installed.")
    print("Run: python -m pip install scikit-learn")
    sys.exit(1)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("model.pkl created successfully")

