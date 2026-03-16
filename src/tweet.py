"""Post tweets with optional media (images/videos) to X/Twitter.

Usage:
    # Text only (dry run by default)
    python src/tweet.py --text "Hello world"

    # With video
    python src/tweet.py --text "Check this out" --media assets/ep4_compare_cfg1.5.mp4

    # Actually send (not dry run)
    python src/tweet.py --text "Hello world" --send

Requires: pip install tweepy python-dotenv
"""

import argparse
import os
import sys
from pathlib import Path

import tweepy
from dotenv import load_dotenv


def get_client():
    """Create authenticated tweepy Client + API (for media uploads)."""
    load_dotenv()

    consumer_key = os.environ.get("X_CONSUMER_KEY")
    consumer_secret = os.environ.get("X_CONSUMER_SECRET")
    access_token = os.environ.get("X_ACCESS_TOKEN")
    access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")

    missing = []
    if not consumer_key:
        missing.append("X_CONSUMER_KEY")
    if not consumer_secret:
        missing.append("X_CONSUMER_SECRET")
    if not access_token:
        missing.append("X_ACCESS_TOKEN")
    if not access_token_secret:
        missing.append("X_ACCESS_TOKEN_SECRET")

    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Set them in .env or generate at https://developer.x.com/en/portal/projects-and-apps")
        sys.exit(1)

    # v2 Client for creating tweets
    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )

    # v1.1 API for media uploads (v2 doesn't support media upload yet)
    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret,
        access_token, access_token_secret,
    )
    api = tweepy.API(auth)

    return client, api


def upload_media(api, media_path):
    """Upload media file and return media_id."""
    path = Path(media_path)
    suffix = path.suffix.lower()

    if suffix in (".mp4", ".mov"):
        print(f"Uploading video: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        media = api.media_upload(
            filename=str(path),
            chunked=True,
            media_category="tweet_video",
        )
    elif suffix in (".png", ".jpg", ".jpeg", ".gif"):
        print(f"Uploading image: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        media = api.media_upload(filename=str(path))
    else:
        print(f"Unsupported media type: {suffix}")
        sys.exit(1)

    print(f"Media uploaded: id={media.media_id_string}")
    return media.media_id_string


def post_tweet(text, media_paths=None, send=False):
    """Post a tweet with optional media."""
    print(f"\n{'=' * 60}")
    print(f"TWEET DRAFT:")
    print(f"{'=' * 60}")
    print(text)
    if media_paths:
        for p in media_paths:
            print(f"  [media] {p}")
    print(f"{'=' * 60}")
    print(f"Characters: {len(text)}/280")

    if len(text) > 280:
        print("ERROR: Tweet exceeds 280 characters!")
        sys.exit(1)

    if not send:
        print("\nDRY RUN — add --send to actually post")
        return

    client, api = get_client()

    media_ids = []
    if media_paths:
        for path in media_paths:
            mid = upload_media(api, path)
            media_ids.append(mid)

    response = client.create_tweet(
        text=text,
        media_ids=media_ids if media_ids else None,
    )

    tweet_id = response.data["id"]
    print(f"\nTweet posted! https://x.com/i/web/status/{tweet_id}")
    return tweet_id


def main():
    parser = argparse.ArgumentParser(description="Post to X/Twitter")
    parser.add_argument("--text", type=str, required=True, help="Tweet text")
    parser.add_argument("--media", type=str, nargs="*", help="Media file paths")
    parser.add_argument("--send", action="store_true", help="Actually send (default is dry run)")
    args = parser.parse_args()

    post_tweet(args.text, args.media, args.send)


if __name__ == "__main__":
    main()
