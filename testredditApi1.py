import praw
import json
from datetime import datetime

# Reddit API Credentials
reddit = praw.Reddit(
    client_id="HA5lGh08tlu2E6y0_LrDPg",
    client_secret="knUDGFR2uwEbHdOPu3sR1bo61R0sXA",
    user_agent="daniel-praw-script",
    username="ai2groupCS4442",
    password="ai2groupCS"
)

# Fetch posts by day using praw
def fetch_posts_by_day(subreddit_name, limit=None):
    subreddit = reddit.subreddit(subreddit_name)
    posts_by_day = {}

    # Fetch submissions (limit is optional)
    for submission in subreddit.new(limit=limit):
        # Convert timestamp to date
        post_date = datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d")
        if post_date not in posts_by_day:
            posts_by_day[post_date] = []

        # Collect relevant submission data
        posts_by_day[post_date].append({
            "title": submission.title,
            "score": submission.score,
            "url": submission.url,
            "created_utc": submission.created_utc,
            "id": submission.id
        })

    return posts_by_day

# Save data to JSON file
def save_posts_to_json(posts_by_day, subreddit_name):
    output_file = f"{subreddit_name}_posts_by_day.json"
    with open(output_file, "w") as file:
        json.dump(posts_by_day, file, indent=4)
    print(f"Posts saved to {output_file}")

# Main function
if __name__ == "__main__":
    subreddit_name = "wallstreetbets"
    # Change `limit=None` to a specific number to limit the posts fetched
    posts_by_day = fetch_posts_by_day(subreddit_name, limit=1000)
    save_posts_to_json(posts_by_day, subreddit_name)
