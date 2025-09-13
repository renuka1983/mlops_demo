import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

# Preprocessing
X = df[["feature1", "feature2"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")