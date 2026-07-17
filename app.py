import os
from datetime import datetime, timezone

import joblib
import numpy as np
import streamlit as st

try:
    import instaloader
except Exception:
    instaloader = None

st.set_page_config(page_title="IG Fake Detector", page_icon="📷", layout="centered")
st.title("Instagram Fake Account Detector — Combined Features")
st.write("Predict if an Instagram account is REAL or FAKE using combined features (15 features).")

MODEL_FILE = "model_combined.joblib"
SCALER_FILE = "scaler_combined.joblib"

def safe_div(a, b):
    return a / b if b != 0 else 0.0

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing file: {MODEL_FILE}")
    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(f"Missing file: {SCALER_FILE}")
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return clf, scaler

try:
    clf, scaler = load_artifacts()
except Exception as e:
    st.error(f"Startup error while loading model files: {e}")
    st.stop()

def predict_combined(feat):
    combined_cols = [
        "username_length", "num_followers", "num_following", "posts_count",
        "is_private", "has_profile_pic", "bio_length", "avg_likes", "avg_comments",
        "account_age_days", "follower_following_ratio", "is_verified",
        "engagement_rate", "gap_follow_ff", "post_frequency"
    ]
    X = np.array([[feat[c] for c in combined_cols]])
    X_scaled = scaler.transform(X)
    pred = clf.predict(X_scaled)[0]
    proba = clf.predict_proba(X_scaled)[0]
    label = "FAKE" if int(pred) == 1 else "REAL"
    return label, float(np.max(proba))

use_username = st.checkbox("Fetch features from Instagram username (auto)", value=True)

L = None
if instaloader is not None:
    try:
        L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            compress_json=False
        )
    except Exception as e:
        st.warning(f"Instaloader setup failed: {e}")

if use_username:
    if instaloader is None or L is None:
        st.warning("Automatic Instagram fetch is unavailable in this deployment. Use manual mode.")
    username = st.text_input("Instagram username (without @)")

    if st.button("Login with saved INSTA_USER (optional)"):
        if L is None:
            st.error("Instaloader is not available.")
        else:
            user = st.secrets.get("INSTA_USER", os.environ.get("INSTA_USER"))
            pw = st.secrets.get("INSTA_PASS", os.environ.get("INSTA_PASS"))
            if not user or not pw:
                st.warning("Set INSTA_USER and INSTA_PASS in Streamlit secrets.")
            else:
                try:
                    L.login(user, pw)
                    st.success(f"Logged in as {user}")
                except Exception as e:
                    st.error(f"Login failed: {e}")

    if st.button("Fetch & Predict"):
        if not username:
            st.warning("Enter username")
        elif L is None:
            st.error("Instagram fetch is unavailable. Switch off auto mode and use manual input.")
        else:
            try:
                profile = instaloader.Profile.from_username(L.context, username)

                num_followers = getattr(profile, "followers", 0) or 0
                num_following = getattr(profile, "followees", 0) or 0
                posts_count = getattr(profile, "mediacount", 0) or 0
                is_private = 1 if getattr(profile, "is_private", False) else 0
                is_verified = 1 if getattr(profile, "is_verified", False) else 0
                has_profile_pic = 0 if not getattr(profile, "profile_pic_url", None) else 1
                bio_text = getattr(profile, "biography", "") or ""
                bio_length = len(bio_text)
                username_len = len(username)

                likes_sum = 0
                comments_sum = 0
                sampled = 0
                oldest_post_ts = None

                try:
                    for post in profile.get_posts():
                        sampled += 1
                        likes_sum += getattr(post, "likes", 0) or 0
                        comments_sum += getattr(post, "comments", 0) or 0
                        created = getattr(post, "date_utc", None)
                        if created:
                            ts = created.replace(tzinfo=timezone.utc).timestamp()
                            if oldest_post_ts is None or ts < oldest_post_ts:
                                oldest_post_ts = ts
                        if sampled >= 30:
                            break
                except Exception as e:
                    st.warning(f"Could not read posts fully: {e}")

                avg_likes = likes_sum / sampled if sampled > 0 else 0.0
                avg_comments = comments_sum / sampled if sampled > 0 else 0.0
                follower_following_ratio = safe_div(num_followers + 1, num_following + 1)
                gap_follow_ff = abs(num_followers - num_following)
                account_age_days = max(
                    1,
                    int((datetime.now(timezone.utc).timestamp() - oldest_post_ts) / 86400)
                ) if oldest_post_ts else 365
                post_frequency = round(posts_count / (account_age_days + 1), 6)
                engagement_rate = round((avg_likes + avg_comments) / (num_followers + 1), 6)

                feat = {
                    "username_length": username_len,
                    "num_followers": num_followers,
                    "num_following": num_following,
                    "posts_count": posts_count,
                    "is_private": is_private,
                    "has_profile_pic": has_profile_pic,
                    "bio_length": bio_length,
                    "avg_likes": avg_likes,
                    "avg_comments": avg_comments,
                    "account_age_days": account_age_days,
                    "follower_following_ratio": round(follower_following_ratio, 3),
                    "is_verified": is_verified,
                    "engagement_rate": engagement_rate,
                    "gap_follow_ff": gap_follow_ff,
                    "post_frequency": post_frequency,
                }

                st.write("Fetched features:", feat)
                label, conf = predict_combined(feat)
                st.subheader(f"Prediction: {label} (Confidence: {conf:.2f})")

            except Exception as e:
                st.error(f"Error fetching username: {e}")

else:
    st.write("Enter manual feature values below.")
    username_len = st.number_input("username_length", 1, 30, value=8)
    num_followers = st.number_input("num_followers", 0, step=1, value=1000)
    num_following = st.number_input("num_following", 0, step=1, value=200)
    posts_count = st.number_input("posts_count", 0, step=1, value=50)
    is_private = st.selectbox("is_private", [0, 1])
    has_profile_pic = st.selectbox("has_profile_pic", [0, 1])
    bio_length = st.number_input("bio_length", 0, value=20)
    avg_likes = st.number_input("avg_likes", 0, value=50)
    avg_comments = st.number_input("avg_comments", 0, value=3)
    account_age_days = st.number_input("account_age_days", 1, value=365)
    is_verified = st.selectbox("is_verified", [0, 1])

    engagement_rate = round((avg_likes + avg_comments) / (num_followers + 1), 6)
    gap_follow_ff = abs(num_followers - num_following)
    post_frequency = round(posts_count / (account_age_days + 1), 6)
    follower_following_ratio = safe_div(num_followers + 1, num_following + 1)

    feat = {
        "username_length": username_len,
        "num_followers": num_followers,
        "num_following": num_following,
        "posts_count": posts_count,
        "is_private": is_private,
        "has_profile_pic": has_profile_pic,
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round(follower_following_ratio, 3),
        "is_verified": is_verified,
        "engagement_rate": engagement_rate,
        "gap_follow_ff": gap_follow_ff,
        "post_frequency": post_frequency,
    }

    if st.button("Predict Manual"):
        label, conf = predict_combined(feat)
        st.subheader(f"Prediction: {label} (Confidence: {conf:.2f})")
