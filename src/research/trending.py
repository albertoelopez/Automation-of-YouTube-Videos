"""Trending topics research - find viral content ideas"""
import json
import httpx
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Iterator
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class TrendingTopic:
    """A trending topic"""
    title: str
    source: str
    category: str = "general"
    score: float = 0.0  # Popularity/virality score
    url: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TrendingTopics:
    """
    Fetch trending topics from various sources.

    Sources:
    - Google Trends (via unofficial API)
    - Reddit (public API)
    - Hacker News
    - YouTube trending (via scraping)
    - Twitter/X trends (requires API)
    """

    def __init__(self, cache_dir: Path | str = "data/trending_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=30.0)

    def get_google_trends(self, geo: str = "US") -> list[TrendingTopic]:
        """Get trending searches from Google Trends"""
        try:
            # Using the daily trends RSS feed
            url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
            response = self.client.get(url)
            response.raise_for_status()

            # Parse RSS/XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)

            topics = []
            for item in root.findall(".//item"):
                title = item.find("title")
                if title is not None and title.text:
                    topics.append(TrendingTopic(
                        title=title.text,
                        source="google_trends",
                        category="trending",
                        score=100.0,
                    ))

            return topics[:20]

        except Exception as e:
            console.print(f"[yellow]Google Trends error: {e}[/yellow]")
            return []

    def get_reddit_trending(
        self,
        subreddits: list[str] | None = None,
        limit: int = 25,
    ) -> list[TrendingTopic]:
        """Get trending posts from Reddit"""
        if subreddits is None:
            subreddits = ["popular", "all"]

        topics = []

        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                headers = {"User-Agent": "VideoAutomation/1.0"}
                response = self.client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()
                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})

                    # Filter out NSFW and low-score posts
                    if post_data.get("over_18"):
                        continue
                    if post_data.get("score", 0) < 100:
                        continue

                    topics.append(TrendingTopic(
                        title=post_data.get("title", ""),
                        source="reddit",
                        category=post_data.get("subreddit", subreddit),
                        score=post_data.get("score", 0),
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        description=post_data.get("selftext", "")[:500],
                    ))

            except Exception as e:
                console.print(f"[yellow]Reddit error ({subreddit}): {e}[/yellow]")

        # Sort by score and deduplicate
        topics.sort(key=lambda t: t.score, reverse=True)
        return topics[:limit]

    def get_hackernews_trending(self, limit: int = 20) -> list[TrendingTopic]:
        """Get top stories from Hacker News"""
        try:
            # Get top story IDs
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = self.client.get(url)
            response.raise_for_status()
            story_ids = response.json()[:limit]

            topics = []
            for story_id in story_ids:
                try:
                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    item_response = self.client.get(item_url)
                    item = item_response.json()

                    if item and item.get("title"):
                        topics.append(TrendingTopic(
                            title=item.get("title", ""),
                            source="hackernews",
                            category="tech",
                            score=item.get("score", 0),
                            url=item.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                        ))
                except Exception:
                    continue

            return topics

        except Exception as e:
            console.print(f"[yellow]Hacker News error: {e}[/yellow]")
            return []

    def get_youtube_trending(
        self,
        category: str = "all",
        region: str = "US",
    ) -> list[TrendingTopic]:
        """
        Get YouTube trending videos.

        Note: This requires YouTube API or scraping.
        For now, returns placeholder suggesting API setup.
        """
        # YouTube trending requires API key
        # This is a placeholder showing how it would work
        console.print("[dim]YouTube trending requires API key setup[/dim]")
        return []

    def get_all_trending(
        self,
        sources: list[str] | None = None,
    ) -> list[TrendingTopic]:
        """Get trending topics from all sources"""
        if sources is None:
            sources = ["google", "reddit", "hackernews"]

        all_topics = []

        if "google" in sources:
            console.print("[dim]Fetching Google Trends...[/dim]")
            all_topics.extend(self.get_google_trends())

        if "reddit" in sources:
            console.print("[dim]Fetching Reddit trending...[/dim]")
            all_topics.extend(self.get_reddit_trending())

        if "hackernews" in sources:
            console.print("[dim]Fetching Hacker News...[/dim]")
            all_topics.extend(self.get_hackernews_trending())

        # Deduplicate by similar titles
        seen_titles = set()
        unique_topics = []
        for topic in all_topics:
            title_lower = topic.title.lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_topics.append(topic)

        return unique_topics

    def search_niche(
        self,
        niche: str,
        subreddits: list[str] | None = None,
    ) -> list[TrendingTopic]:
        """Search for trending topics in a specific niche"""
        # Map niches to subreddits
        niche_subreddits = {
            "tech": ["technology", "programming", "gadgets", "apple", "android"],
            "gaming": ["gaming", "games", "pcgaming", "nintendo", "playstation"],
            "finance": ["finance", "investing", "stocks", "cryptocurrency", "personalfinance"],
            "fitness": ["fitness", "gym", "bodybuilding", "running", "nutrition"],
            "cooking": ["cooking", "food", "recipes", "MealPrepSunday", "EatCheapAndHealthy"],
            "science": ["science", "space", "physics", "biology", "chemistry"],
            "movies": ["movies", "television", "netflix", "Marvel", "StarWars"],
            "music": ["music", "hiphopheads", "indieheads", "Metal", "EDM"],
            "travel": ["travel", "backpacking", "solotravel", "roadtrip"],
            "diy": ["DIY", "woodworking", "crafts", "homeimprovement"],
        }

        if subreddits is None:
            subreddits = niche_subreddits.get(niche.lower(), [niche])

        return self.get_reddit_trending(subreddits=subreddits, limit=30)

    def print_trending(self, topics: list[TrendingTopic], limit: int = 20):
        """Print trending topics in a table"""
        table = Table(title="Trending Topics")
        table.add_column("#", style="dim")
        table.add_column("Title")
        table.add_column("Source")
        table.add_column("Category")
        table.add_column("Score", justify="right")

        for i, topic in enumerate(topics[:limit], 1):
            table.add_row(
                str(i),
                topic.title[:60] + "..." if len(topic.title) > 60 else topic.title,
                topic.source,
                topic.category,
                f"{topic.score:,.0f}" if topic.score > 0 else "-",
            )

        console.print(table)

    def save_cache(self, topics: list[TrendingTopic], name: str = "trending"):
        """Save topics to cache"""
        cache_file = self.cache_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        data = []
        for topic in topics:
            data.append({
                "title": topic.title,
                "source": topic.source,
                "category": topic.category,
                "score": topic.score,
                "url": topic.url,
                "description": topic.description,
                "tags": topic.tags,
                "timestamp": topic.timestamp.isoformat() if topic.timestamp else None,
            })

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        return cache_file


def get_trending_topics(
    sources: list[str] | None = None,
    niche: str | None = None,
) -> list[TrendingTopic]:
    """Convenience function to get trending topics"""
    trending = TrendingTopics()

    if niche:
        return trending.search_niche(niche)
    else:
        return trending.get_all_trending(sources)
