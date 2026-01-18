"""Unit tests for research module"""
import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.research.trending import TrendingTopics, TrendingTopic, get_trending_topics
from src.research.ideas import IdeaGenerator, ContentIdea


class TestTrendingTopic:
    """Tests for TrendingTopic dataclass"""

    def test_create_trending_topic(self):
        """Test creating a trending topic"""
        topic = TrendingTopic(
            title="Test Topic",
            source="reddit",
            category="tech",
            score=1000,
        )

        assert topic.title == "Test Topic"
        assert topic.source == "reddit"
        assert topic.category == "tech"
        assert topic.score == 1000
        assert topic.timestamp is not None

    def test_trending_topic_defaults(self):
        """Test default values"""
        topic = TrendingTopic(title="Test", source="test")

        assert topic.category == "general"
        assert topic.score == 0.0
        assert topic.url == ""
        assert topic.description == ""
        assert topic.tags == []

    def test_trending_topic_with_timestamp(self):
        """Test with custom timestamp"""
        ts = datetime(2024, 1, 1, 12, 0)
        topic = TrendingTopic(title="Test", source="test", timestamp=ts)

        assert topic.timestamp == ts


class TestTrendingTopics:
    """Tests for TrendingTopics class"""

    @pytest.fixture
    def trending(self, temp_dir):
        """Create TrendingTopics instance with temp cache dir"""
        return TrendingTopics(cache_dir=temp_dir / "cache")

    def test_init_creates_cache_dir(self, temp_dir):
        """Test that init creates cache directory"""
        cache_dir = temp_dir / "trending_cache"
        trending = TrendingTopics(cache_dir=cache_dir)

        assert cache_dir.exists()

    @pytest.mark.slow
    def test_get_reddit_trending_mock(self, trending):
        """Test Reddit trending with mocked response"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Test Post 1",
                            "subreddit": "technology",
                            "score": 5000,
                            "permalink": "/r/technology/123",
                            "selftext": "Test content",
                            "over_18": False,
                        }
                    },
                    {
                        "data": {
                            "title": "Test Post 2",
                            "subreddit": "programming",
                            "score": 3000,
                            "permalink": "/r/programming/456",
                            "selftext": "",
                            "over_18": False,
                        }
                    },
                    {
                        "data": {
                            "title": "NSFW Post",
                            "subreddit": "other",
                            "score": 10000,
                            "permalink": "/r/other/789",
                            "over_18": True,
                        }
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(trending.client, "get", return_value=mock_response):
            topics = trending.get_reddit_trending(subreddits=["technology"], limit=10)

        # Should filter out NSFW
        assert len(topics) == 2
        assert topics[0].title == "Test Post 1"
        assert topics[0].source == "reddit"
        assert topics[0].score == 5000

    def test_get_reddit_filters_low_score(self, trending):
        """Test that low score posts are filtered"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {"data": {"title": "High Score", "score": 500, "over_18": False, "subreddit": "test", "permalink": "/1"}},
                    {"data": {"title": "Low Score", "score": 50, "over_18": False, "subreddit": "test", "permalink": "/2"}},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(trending.client, "get", return_value=mock_response):
            topics = trending.get_reddit_trending(subreddits=["test"])

        assert len(topics) == 1
        assert topics[0].title == "High Score"

    def test_get_hackernews_trending_mock(self, trending):
        """Test Hacker News trending with mocked response"""
        # Mock the story IDs response
        mock_ids_response = MagicMock()
        mock_ids_response.json.return_value = [1, 2, 3]
        mock_ids_response.raise_for_status = MagicMock()

        # Mock individual story responses
        mock_story_response = MagicMock()
        mock_story_response.json.return_value = {
            "title": "Test HN Story",
            "score": 200,
            "url": "https://example.com",
        }

        with patch.object(trending.client, "get") as mock_get:
            mock_get.side_effect = [mock_ids_response] + [mock_story_response] * 3
            topics = trending.get_hackernews_trending(limit=3)

        assert len(topics) == 3
        assert topics[0].source == "hackernews"
        assert topics[0].category == "tech"

    def test_search_niche(self, trending):
        """Test niche search maps to correct subreddits"""
        with patch.object(trending, "get_reddit_trending") as mock_reddit:
            mock_reddit.return_value = []
            trending.search_niche("tech")

        mock_reddit.assert_called_once()
        call_args = mock_reddit.call_args
        subreddits = call_args[1]["subreddits"]
        assert "technology" in subreddits
        assert "programming" in subreddits

    def test_save_cache(self, trending, temp_dir):
        """Test saving topics to cache"""
        topics = [
            TrendingTopic(title="Test 1", source="test", score=100),
            TrendingTopic(title="Test 2", source="test", score=200),
        ]

        cache_file = trending.save_cache(topics, name="test_cache")

        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["title"] == "Test 1"

    def test_get_all_trending_combines_sources(self, trending):
        """Test that get_all_trending combines multiple sources"""
        with patch.object(trending, "get_google_trends") as mock_google, \
             patch.object(trending, "get_reddit_trending") as mock_reddit, \
             patch.object(trending, "get_hackernews_trending") as mock_hn:

            mock_google.return_value = [TrendingTopic(title="Google Topic", source="google")]
            mock_reddit.return_value = [TrendingTopic(title="Reddit Topic", source="reddit")]
            mock_hn.return_value = [TrendingTopic(title="HN Topic", source="hackernews")]

            topics = trending.get_all_trending()

        assert len(topics) == 3
        sources = {t.source for t in topics}
        assert sources == {"google", "reddit", "hackernews"}

    def test_deduplication(self, trending):
        """Test that duplicate titles are removed"""
        with patch.object(trending, "get_google_trends") as mock_google, \
             patch.object(trending, "get_reddit_trending") as mock_reddit, \
             patch.object(trending, "get_hackernews_trending") as mock_hn:

            mock_google.return_value = [TrendingTopic(title="Same Topic", source="google")]
            mock_reddit.return_value = [TrendingTopic(title="same topic", source="reddit")]  # Same, different case
            mock_hn.return_value = [TrendingTopic(title="Different Topic", source="hackernews")]

            topics = trending.get_all_trending()

        assert len(topics) == 2


class TestContentIdea:
    """Tests for ContentIdea dataclass"""

    def test_create_content_idea(self):
        """Test creating a content idea"""
        idea = ContentIdea(
            title="Test Title",
            hook="Amazing hook!",
            outline=["Point 1", "Point 2"],
            target_audience="Developers",
            estimated_duration=30,
            difficulty="easy",
            viral_potential="high",
            tags=["test", "video"],
        )

        assert idea.title == "Test Title"
        assert idea.hook == "Amazing hook!"
        assert len(idea.outline) == 2
        assert idea.difficulty == "easy"
        assert idea.viral_potential == "high"


class TestIdeaGenerator:
    """Tests for IdeaGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create IdeaGenerator with mocked LLM"""
        return IdeaGenerator()

    def test_generate_idea_with_mock_llm(self, generator, sample_content_idea):
        """Test idea generation with mocked LLM"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps(sample_content_idea)

        generator.llm_client = mock_llm
        idea = generator.generate_idea("Python tips")

        assert idea is not None
        assert idea.title == sample_content_idea["title"]
        assert idea.hook == sample_content_idea["hook"]
        assert idea.viral_potential == "high"

    def test_generate_idea_llm_unavailable(self, generator):
        """Test behavior when LLM is unavailable"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False

        generator.llm_client = mock_llm
        idea = generator.generate_idea("Test topic")

        assert idea is None

    def test_generate_idea_handles_json_in_markdown(self, generator, sample_content_idea):
        """Test parsing JSON wrapped in markdown code blocks"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = f"```json\n{json.dumps(sample_content_idea)}\n```"

        generator.llm_client = mock_llm
        idea = generator.generate_idea("Test topic")

        assert idea is not None
        assert idea.title == sample_content_idea["title"]

    def test_generate_idea_handles_invalid_json(self, generator):
        """Test graceful handling of invalid JSON"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = "This is not valid JSON"

        generator.llm_client = mock_llm
        idea = generator.generate_idea("Test topic")

        assert idea is None

    def test_generate_ideas_batch(self, generator, sample_content_idea):
        """Test batch idea generation"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps(sample_content_idea)

        generator.llm_client = mock_llm
        ideas = generator.generate_ideas_batch("Python", count=3)

        assert len(ideas) == 3
        assert mock_llm.generate.call_count == 3

    def test_generate_from_trending(self, generator, sample_content_idea):
        """Test generating ideas from trending topics"""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps(sample_content_idea)

        generator.llm_client = mock_llm

        trending_topics = [
            TrendingTopic(title="AI News", source="test"),
            TrendingTopic(title="Python Update", source="test"),
        ]

        ideas = generator.generate_from_trending(trending_topics, count=2)

        assert len(ideas) == 2


class TestGetTrendingTopicsFunction:
    """Tests for the convenience function"""

    def test_get_trending_topics_default(self):
        """Test default behavior"""
        with patch("src.research.trending.TrendingTopics") as mock_class:
            mock_instance = MagicMock()
            mock_instance.get_all_trending.return_value = []
            mock_class.return_value = mock_instance

            get_trending_topics()

            mock_instance.get_all_trending.assert_called_once()

    def test_get_trending_topics_with_niche(self):
        """Test with niche parameter"""
        with patch("src.research.trending.TrendingTopics") as mock_class:
            mock_instance = MagicMock()
            mock_instance.search_niche.return_value = []
            mock_class.return_value = mock_instance

            get_trending_topics(niche="tech")

            mock_instance.search_niche.assert_called_once_with("tech")
