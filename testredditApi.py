import praw

# Reddit API Credentials
reddit = praw.Reddit(
    client_id="HA5lGh08tlu2E6y0_LrDPg",
    client_secret="knUDGFR2uwEbHdOPu3sR1bo61R0sXA",
    user_agent="Daniel Esemezie",
    username = "ai2groupCS4442",
    password = "ai2groupCS"
)

# Fetching Posts from a Subreddit
subreddit = reddit.subreddit("wallstreetbets")
for post in subreddit.new(limit=10):
    print(f"Title: {post.title}")
    print(f"Score: {post.score}")
    print(f"URL: {post.url}")
    print("-" * 40)