# scraper_instagram.py
"""
Instaloader-based scraper to collect Instagram profile features into instagram_dataset.csv

Usage:
  - Optionally set env vars INSTA_USER and INSTA_PASS to login and access private accounts you follow.
  - Put usernames (one per line) in usernames.txt or pass a small list programmatically.
  - Run: python scraper_instagram.py
"""

import os, csv, time
import instaloader
from datetime import datetime
from math import log1p

SESSION_USER = os.environ.get("INSTA_USER")
SESSION_PASS = os.environ.get("INSTA_PASS")
OUT_CSV = "instagram_dataset.csv"
USERNAMES_FILE = "usernames.txt"  # one username per line

L = instaloader.Instaloader(download_pictures=False, download_videos=False,
                            download_comments=False, save_metadata=False, compress_json=False)

if SESSION_USER and SESSION_PASS:
    try:
        L.login(SESSION_USER, SESSION_PASS)
        print("[scraper] logged in as", SESSION_USER)
    except Exception as e:
        print("[scraper] login failed:", e)

def extract_profile_features(profile, max_posts=20):
    """Return dict of features for a profile (Instaloader Profile object)."""
    username = profile.username
    num_followers = int(getattr(profile, "followers", 0) or 0)
    num_following = int(getattr(profile, "followees", 0) or 0)
    posts_count = int(getattr(profile, "mediacount", 0) or 0)
    is_private = 1 if getattr(profile, "is_private", False) else 0
    is_verified = 1 if getattr(profile, "is_verified", False) else 0
    has_profile_pic = 0 if (getattr(profile, "profile_pic_url", None) in (None, "")) else 1
    bio_length = len(getattr(profile, "biography", "") or "")

    likes_sum = 0
    comments_sum = 0
    sampled = 0
    try:
        for post in profile.get_posts():
            likes_sum += getattr(post, "likes", 0) or 0
            comments_sum += getattr(post, "comments", 0) or 0
            sampled += 1
            if sampled >= max_posts:
                break
    except Exception:
        sampled = 0

    avg_likes = (likes_sum / sampled) if sampled > 0 else 0.0
    avg_comments = (comments_sum / sampled) if sampled > 0 else 0.0

    # approximate account age by first_available_post date if possible
    # here fallback to 365 days if unknown
    account_age_days = 365.0
    try:
        # attempt to compute earliest post date
        first_date = None
        for post in profile.get_posts():
            first_date = post.date_utc
            break
        if first_date:
            account_age_days = (datetime.utcnow() - first_date).days
            if account_age_days <= 0:
                account_age_days = 365.0
    except Exception:
        pass

    # derived features
    follower_following_ratio = (num_followers + 1) / (num_following + 1)

    return {
        "username": username,
        "num_followers": num_followers,
        "num_following": num_following,
        "posts_count": posts_count,
        "is_private": is_private,
        "is_verified": is_verified,
        "has_profile_pic": has_profile_pic,
        "bio_length": bio_length,
        "avg_likes": round(avg_likes, 4),
        "avg_comments": round(avg_comments, 4),
        "account_age_days": account_age_days,
        "follower_following_ratio": round(follower_following_ratio, 4),
        "posts_sampled": sampled
    }

def load_usernames():
    if not os.path.exists(USERNAMES_FILE):
        print(f"[scraper] create {USERNAMES_FILE} with one username per line, then run scraper.")
        return []
    with open(USERNAMES_FILE, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def append_to_csv(rows, filename=OUT_CSV):
    header = ["username","num_followers","num_following","posts_count","is_private","is_verified","has_profile_pic",
              "bio_length","avg_likes","avg_comments","account_age_days","follower_following_ratio","posts_sampled","label"]
    exists = os.path.exists(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    names = load_usernames()
    if not names:
        return
    results = []
    for name in names:
        try:
            profile = instaloader.Profile.from_username(L.context, name)
            feats = extract_profile_features(profile, max_posts=20)
            # label unknown — set label = -1 to mark unlabeled (you must label later)
            feats["label"] = -1
            results.append(feats)
            print(f"[scraper] fetched {name} | followers={feats['num_followers']} posts={feats['posts_count']} verified={feats['is_verified']}")
        except Exception as e:
            print(f"[scraper] error for {name}: {e}")
        time.sleep(2)  # polite delay
    if results:
        append_to_csv(results)
        print(f"[scraper] appended {len(results)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
