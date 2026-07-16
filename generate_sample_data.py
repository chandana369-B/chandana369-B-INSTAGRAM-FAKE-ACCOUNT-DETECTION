# generate_sample_dataset.py
"""
Generates a synthetic instagram_data.csv with a mix of real/verified/ fake accounts.
"""
import csv
import random
import math

OUT = "instagram_sample.csv"
rows = []
random.seed(0)

def make_real(verified=False):
    if verified:
        followers = int(10**random.uniform(5,8))  # large
        following = int(10**random.uniform(1,3))
        posts = int(random.uniform(100,2000))
        avg_likes = int(followers * random.uniform(0.02, 0.6))
    else:
        followers = int(10**random.uniform(2,5))
        following = int(10**random.uniform(1,4))
        posts = int(random.uniform(20,800))
        avg_likes = int(followers * random.uniform(0.01, 0.2))
    avg_comments = max(0, int(avg_likes * random.uniform(0.01, 0.05)))
    bio_length = random.randint(20,200)
    account_age_days = random.randint(200,3000)
    return {
        "username_length": random.randint(4,16),
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": 0,
        "has_profile_pic": 1,
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers+1)/(following+1),4),
        "is_verified": 1 if verified else 0,
        "label": 0
    }

def make_fake():
    followers = int(10**random.uniform(0.8,3))
    following = int(10**random.uniform(2.5,4.5))
    posts = int(random.uniform(0,30))
    avg_likes = int(random.uniform(0,20))
    avg_comments = int(random.uniform(0,5))
    bio_length = random.choice([0, random.randint(1,30)])
    account_age_days = random.randint(1,400)
    return {
        "username_length": random.randint(4,18),
        "num_followers": followers,
        "num_following": following,
        "posts_count": posts,
        "is_private": random.choice([0,1]),
        "has_profile_pic": random.choice([0,1]),
        "bio_length": bio_length,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "account_age_days": account_age_days,
        "follower_following_ratio": round((followers+1)/(following+1),4),
        "is_verified": 0,
        "label": 1
    }

# create 300 rows (mix)
for _ in range(120):
    rows.append(make_real(verified=False))
for _ in range(30):
    rows.append(make_real(verified=True))
for _ in range(150):
    rows.append(make_fake())

# shuffle and write
random.shuffle(rows)
fields = list(rows[0].keys())
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[sample] written {OUT} with {len(rows)} rows")
