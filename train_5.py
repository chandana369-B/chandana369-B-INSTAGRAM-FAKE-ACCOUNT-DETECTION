# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("instagram_ultra_dataset.csv")
print("Dataset loaded:", df.shape, "rows")

# ----------------------------
# Feature Groups
# ----------------------------
ratio_features = ["num_followers", "num_following"]

engagement_features = [
    "num_followers",
    "num_following",
    "posts_count",
    "avg_likes",
    "avg_comments",
    "follower_following_ratio"
]

verified_features = [
    "num_followers",
    "num_following",
    "posts_count",
    "is_verified"
]

# Combined = all features including high-impact features
combined_features = [
    "username_length",
    "num_followers",
    "num_following",
    "posts_count",
    "is_private",
    "has_profile_pic",
    "bio_length",
    "avg_likes",
    "avg_comments",
    "account_age_days",
    "follower_following_ratio",
    "is_verified",
    "engagement_rate",
    "growth_score",
    "username_suspicious_score"
]

target = "label"

# ----------------------------------
# Function to train, scale, evaluate, save model
# ----------------------------------
def train_and_save_model(name, feature_list):
    print("\n===============================")
    print(" TRAINING:", name)
    print("===============================")

    X = df[feature_list].values
    y = df[target].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy = {acc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save model and scaler
    model_file = f"model_{name}.joblib"
    scaler_file = f"scaler_{name}.joblib"
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    print("Saved:", model_file, "and", scaler_file)
    print("-----------------------------------")

# ----------------------------------
# Train all models
# ----------------------------------
train_and_save_model("ratio", ratio_features)
train_and_save_model("engagement", engagement_features)
train_and_save_model("verified", verified_features)
train_and_save_model("combined", combined_features)

print("\nDONE: All models trained and saved.")
