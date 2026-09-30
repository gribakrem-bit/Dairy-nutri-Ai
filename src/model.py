import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/dairy_products.csv")

# Features used by the model
features = [
    "Energy_kcal",
    "Fat_g",
    "Protein_g",
    "Carbohydrate_g",
    "Calcium_mg",
    "Sodium_mg",
    "Water_g"
]

X = df[features]
y = df["Category"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("DairyNutri AI")
print("-------------")
print(f"Model accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)
