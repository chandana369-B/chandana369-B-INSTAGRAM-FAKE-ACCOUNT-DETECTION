"""
Ultra-Realistic 5000-row Instagram Dataset Generator
Includes high-impact features: engagement_rate, growth_score, username_suspicious_score
"""

import csv
import random

OUT = "instagram_ultra_dataset.csv"
rows = []
random.seed(42)

# -----------------------
# Helper functions
# -----------------------
def calc_engagement(followers, avg_likes, avg_comments):
    if followers == 0:
        return 0.0
    return round((avg_likes + avg_comments) / followers, 4)

def generate_growth_score(account_age_days, followers):
    # growth = followers / account_age_days with some noise
    if account_age_days <= 0:
        return 0.0
    score = followers / account_age_days
    return round(score * random.uniform(0.8, 1.2), 2)

def username_suspicious(username_length):
    # longer usernames or random patterns more suspicious
    score = username_length / 15.0
    return round(min(score, 1.0), 2)

# -----------------------
# Account generators
# -----------------------
def make_verified():
    followers = int(10**random.uniform(6, 8.5))  # 1M–300M
    following = random.randint(0, 100)
    posts = random.randint(1, 1_000_000)
    bio_length = random.randint(50, 250)

    avg_likes = int(followers * random.uniform(0.01, 0.15))
    avg_comments = int(avg_likes * random.uniform(0.005, 0.02))
    account_age_days = random.randint(800, 4500)
    username_len = random.randint(3, 20)

    eng_rate = calc_engagement(followers, avg_likes, avg_comments)
    growth = generate_growth_score(account_age_days, followers)
    suspicious_score = username_suspicious(username_len)

    return {
        "username_length": username_len,
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": 0,
        "has_profile_pic": 1,
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers + 1) / (following + 1), 3),
        "is_verified": 1,
        "engagement_rate": eng_rate,
        "growth_score": growth,
        "username_suspicious_score": suspicious_score,
        "label": 0
    }

def make_public_real():
    followers = random.randint(200, 1_000_000)
    following = random.randint(50, 3000)
    posts = random.randint(10, 300)
    bio_length = random.randint(20, 200)
    account_age_days = random.randint(300, 4000)
    username_len = random.randint(3, 18)

    avg_likes = int(followers * random.uniform(0.005, 0.2))
    avg_comments = int(avg_likes * random.uniform(0.01, 0.05))

    eng_rate = calc_engagement(followers, avg_likes, avg_comments)
    growth = generate_growth_score(account_age_days, followers)
    suspicious_score = username_suspicious(username_len)

    return {
        "username_length": username_len,
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": 0,
        "has_profile_pic": 1,
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers + 1) / (following + 1), 3),
        "is_verified": 0,
        "engagement_rate": eng_rate,
        "growth_score": growth,
        "username_suspicious_score": suspicious_score,
        "label": 0
    }

def make_private_real():
    followers = random.randint(30, 30_000)
    following = random.randint(30, 300)
    posts = random.randint(0, 500)
    bio_length = random.randint(10, 150)
    account_age_days = random.randint(50, 3000)
    username_len = random.randint(3, 18)

    eng_rate = 0.0
    growth = generate_growth_score(account_age_days, followers)
    suspicious_score = username_suspicious(username_len)

    return {
        "username_length": username_len,
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": 1,
        "has_profile_pic": random.choice([0, 1]),
        "bio_length": bio_length,
        "avg_likes": 0,
        "avg_comments": 0,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers + 1) / (following + 1), 3),
        "is_verified": 0,
        "engagement_rate": eng_rate,
        "growth_score": growth,
        "username_suspicious_score": suspicious_score,
        "label": 0
    }

def make_fake():
    followers = random.randint(0, 20)
    following = random.randint(0, 30)
    posts = random.randint(0, 10)
    bio_length = random.randint(0, 50)
    account_age_days = random.randint(10, 100)
    username_len = random.randint(5, 20)

    avg_likes = random.randint(0, 5)
    avg_comments = random.randint(0, 2)
    eng_rate = calc_engagement(followers, avg_likes, avg_comments)
    growth = generate_growth_score(account_age_days, followers)
    suspicious_score = username_suspicious(username_len) + random.uniform(0.1, 0.3)
    suspicious_score = round(min(suspicious_score, 1.0), 2)

    return {
        "username_length": username_len,
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": random.choice([0, 1]),
        "has_profile_pic": random.choice([0, 1]),
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers + 1) / (following + 1), 3),
        "is_verified": 0,
        "engagement_rate": eng_rate,
        "growth_score": growth,
        "username_suspicious_score": suspicious_score,
        "label": 1
    }

# -----------------------
# Generate dataset
# -----------------------
COUNT_VERIFIED = 600
COUNT_PUBLIC_REAL = 1400
COUNT_PRIVATE_REAL = 1000
COUNT_FAKE = 2000

for _ in range(COUNT_VERIFIED):
    rows.append(make_verified())
for _ in range(COUNT_PUBLIC_REAL):
    rows.append(make_public_real())
for _ in range(COUNT_PRIVATE_REAL):
    rows.append(make_private_real())
for _ in range(COUNT_FAKE):
    rows.append(make_fake())

random.shuffle(rows)

fields = list(rows[0].keys())
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[DONE] Created {len(rows)} rows → {OUT}")
