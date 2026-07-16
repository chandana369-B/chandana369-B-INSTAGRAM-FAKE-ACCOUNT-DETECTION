# train_all.py
"""
Train four models using instagram CSV. Choose csv path in DATA_CSV.
Saves models and scalers for use in the Streamlit app.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

DATA_CSV = "instagram_sample.csv"   # or instagram_dataset.csv from scraper
RANDOM_STATE = 42

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"{DATA_CSV} not found. Generate sample or scrape data first.")

df = pd.read_csv(DATA_CSV)
# drop rows with missing label
df = df.dropna(subset=["label"])

# make sure label is int
df["label"] = df["label"].astype(int)

# --- Feature sets ---
# Ratio model: followers & following
X_ratio = df[["num_followers","num_following"]].fillna(0)
# Engagement: followers, following, posts_count, avg_likes, avg_comments, follower_following_ratio
X_eng = df[["num_followers","num_following","posts_count","avg_likes","avg_comments","follower_following_ratio"]].fillna(0)
# Verified: include is_verified (and simple counts)
X_verified = df[["num_followers","num_following","posts_count","is_verified"]].fillna(0)
# Combined: everything available (except username)
all_features = [c for c in df.columns if c not in ("username","label")]
X_comb = df[all_features].fillna(0)

y = df["label"]

def train_and_save(X, name):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    # calibrate
    clf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)
    # save
    model_name = f"model_{name}.joblib"
    scaler_name = f"scaler_{name}.joblib"
    joblib.dump(clf, model_name)
    joblib.dump(scaler, scaler_name)
    print(f"[train] Saved {model_name} and {scaler_name}")

print("[train] training ratio model...")
train_and_save(X_ratio, "ratio")

print("[train] training engagement model...")
train_and_save(X_eng, "engagement")

print("[train] training verified model...")
train_and_save(X_verified, "verified")

print("[train] training combined model...")
train_and_save(X_comb, "combined")

print("[train] All models trained.")
