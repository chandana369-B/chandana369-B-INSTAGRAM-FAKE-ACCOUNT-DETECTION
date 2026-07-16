# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("dataset.csv")

FEATURES = [
    "username_length","num_followers","num_following","posts_count",
    "is_private","has_profile_pic","bio_length","avg_likes",
    "avg_comments","account_age_days","follower_following_ratio"
]

X = df[FEATURES].values
y = df["is_fake"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_s, y_train)

test_acc = clf.score(X_test_s, y_test)
print("Test accuracy:", test_acc)

joblib.dump(clf, "rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved rf_model.joblib and scaler.joblib")