"""Unit tests for A/B testing module"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.optimization.ab_testing import (
    ABTestManager,
    ABTest,
    Variant,
    _norm_cdf,
)


class TestVariant:
    """Tests for Variant dataclass"""

    def test_create_variant(self):
        """Test creating a variant"""
        variant = Variant(
            id="test_a",
            name="Variant A",
            value="Test Title",
        )

        assert variant.id == "test_a"
        assert variant.name == "Variant A"
        assert variant.impressions == 0
        assert variant.clicks == 0

    def test_ctr_zero_impressions(self):
        """Test CTR with zero impressions"""
        variant = Variant(id="test", name="Test", value="Test")

        assert variant.ctr == 0.0

    def test_ctr_calculation(self):
        """Test CTR calculation"""
        variant = Variant(
            id="test",
            name="Test",
            value="Test",
            impressions=1000,
            clicks=50,
        )

        assert variant.ctr == 5.0  # 50/1000 * 100 = 5%

    def test_engagement_rate_zero_impressions(self):
        """Test engagement rate with zero impressions"""
        variant = Variant(id="test", name="Test", value="Test")

        assert variant.engagement_rate == 0.0

    def test_engagement_rate_calculation(self):
        """Test engagement rate calculation"""
        variant = Variant(
            id="test",
            name="Test",
            value="Test",
            impressions=1000,
            likes=30,
            comments=10,
            shares=10,
        )

        # (30 + 10 + 10) / 1000 * 100 = 5%
        assert variant.engagement_rate == 5.0


class TestABTest:
    """Tests for ABTest dataclass"""

    @pytest.fixture
    def sample_test(self):
        """Create a sample A/B test"""
        return ABTest(
            id="test123",
            video_id="video456",
            test_type="title",
            variant_a=Variant(id="test123_a", name="Variant A", value="Title A"),
            variant_b=Variant(id="test123_b", name="Variant B", value="Title B"),
        )

    def test_create_test(self, sample_test):
        """Test creating an A/B test"""
        assert sample_test.id == "test123"
        assert sample_test.video_id == "video456"
        assert sample_test.test_type == "title"
        assert sample_test.status == "active"
        assert sample_test.winner is None

    def test_get_winner_insufficient_impressions(self, sample_test):
        """Test winner detection with insufficient impressions"""
        sample_test.variant_a.impressions = 100
        sample_test.variant_b.impressions = 100

        winner, confidence = sample_test.get_winner()

        assert winner is None
        assert confidence == 0.0

    def test_get_winner_significant_difference(self, sample_test):
        """Test winner detection with significant difference"""
        # Set up clear winner with statistical significance
        sample_test.variant_a.impressions = 10000
        sample_test.variant_a.clicks = 500  # 5% CTR

        sample_test.variant_b.impressions = 10000
        sample_test.variant_b.clicks = 300  # 3% CTR

        sample_test.min_impressions = 1000

        winner, confidence = sample_test.get_winner()

        assert winner is not None
        assert winner.name == "Variant A"
        assert confidence > 0.95

    def test_get_winner_no_significant_difference(self, sample_test):
        """Test when there's no significant difference"""
        sample_test.variant_a.impressions = 1000
        sample_test.variant_a.clicks = 50  # 5%

        sample_test.variant_b.impressions = 1000
        sample_test.variant_b.clicks = 52  # 5.2%

        sample_test.min_impressions = 500

        winner, confidence = sample_test.get_winner()

        # Should not declare winner with such small difference
        assert confidence < 0.95


class TestABTestManager:
    """Tests for ABTestManager class"""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create manager with temp database"""
        return ABTestManager(db_path=temp_dir / "ab_tests.db")

    def test_init_creates_db(self, temp_dir):
        """Test that init creates database"""
        db_path = temp_dir / "test.db"
        manager = ABTestManager(db_path=db_path)

        assert db_path.exists()

    def test_create_test(self, manager):
        """Test creating a new test"""
        test = manager.create_test(
            video_id="vid123",
            test_type="title",
            variant_a_value="Title A",
            variant_b_value="Title B",
        )

        assert test is not None
        assert test.video_id == "vid123"
        assert test.test_type == "title"
        assert test.variant_a.value == "Title A"
        assert test.variant_b.value == "Title B"

    def test_get_test(self, manager):
        """Test retrieving a test"""
        created = manager.create_test(
            video_id="vid123",
            test_type="thumbnail",
            variant_a_value="/path/a.png",
            variant_b_value="/path/b.png",
        )

        retrieved = manager.get_test(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.video_id == "vid123"

    def test_get_nonexistent_test(self, manager):
        """Test retrieving non-existent test"""
        result = manager.get_test("nonexistent")

        assert result is None

    def test_list_tests(self, manager):
        """Test listing tests"""
        manager.create_test("vid1", "title", "A1", "B1")
        manager.create_test("vid2", "title", "A2", "B2")
        manager.create_test("vid3", "thumbnail", "A3", "B3")

        all_tests = manager.list_tests()
        assert len(all_tests) == 3

    def test_list_tests_filter_by_status(self, manager):
        """Test filtering by status"""
        test1 = manager.create_test("vid1", "title", "A1", "B1")
        test2 = manager.create_test("vid2", "title", "A2", "B2")

        manager.end_test(test1.id)

        active_tests = manager.list_tests(status="active")
        completed_tests = manager.list_tests(status="completed")

        assert len(active_tests) == 1
        assert len(completed_tests) == 1

    def test_list_tests_filter_by_video(self, manager):
        """Test filtering by video ID"""
        manager.create_test("vid1", "title", "A1", "B1")
        manager.create_test("vid1", "thumbnail", "A2", "B2")
        manager.create_test("vid2", "title", "A3", "B3")

        vid1_tests = manager.list_tests(video_id="vid1")

        assert len(vid1_tests) == 2

    def test_record_impression(self, manager):
        """Test recording impressions"""
        test = manager.create_test("vid1", "title", "A", "B")

        manager.record_impression(test.id, "a")
        manager.record_impression(test.id, "a")
        manager.record_impression(test.id, "b")

        updated = manager.get_test(test.id)

        assert updated.variant_a.impressions == 2
        assert updated.variant_b.impressions == 1

    def test_record_click(self, manager):
        """Test recording clicks"""
        test = manager.create_test("vid1", "title", "A", "B")

        manager.record_click(test.id, "a")
        manager.record_click(test.id, "b")
        manager.record_click(test.id, "b")

        updated = manager.get_test(test.id)

        assert updated.variant_a.clicks == 1
        assert updated.variant_b.clicks == 2

    def test_select_variant_exploration(self, manager):
        """Test variant selection includes exploration"""
        test = manager.create_test("vid1", "title", "A", "B")

        # Run many selections to verify both variants get picked
        selections = {"a": 0, "b": 0}
        for _ in range(100):
            variant = manager.select_variant(test.id)
            selections[variant] += 1

        # Both should be selected at least sometimes (epsilon-greedy)
        assert selections["a"] > 0
        assert selections["b"] > 0

    def test_select_variant_inactive_test(self, manager):
        """Test variant selection for inactive test"""
        test = manager.create_test("vid1", "title", "A", "B")
        manager.end_test(test.id)

        variant = manager.select_variant(test.id)

        # Should default to "a" for inactive tests
        assert variant == "a"

    def test_check_winner(self, manager):
        """Test checking for winner"""
        test = manager.create_test("vid1", "title", "A", "B", min_impressions=100)

        # Add data to variant A
        for _ in range(200):
            manager.record_impression(test.id, "a")
        for _ in range(20):
            manager.record_click(test.id, "a")

        # Add data to variant B
        for _ in range(200):
            manager.record_impression(test.id, "b")
        for _ in range(5):
            manager.record_click(test.id, "b")

        has_winner, winner_name, confidence = manager.check_winner(test.id)

        assert has_winner is True
        assert winner_name == "Variant A"
        assert confidence > 0.95

    def test_end_test(self, manager):
        """Test ending a test"""
        test = manager.create_test("vid1", "title", "A", "B")

        manager.end_test(test.id)

        updated = manager.get_test(test.id)

        assert updated.status == "completed"
        assert updated.ended_at is not None

    def test_end_test_with_forced_winner(self, manager):
        """Test ending test with forced winner"""
        test = manager.create_test("vid1", "title", "A", "B")

        manager.end_test(test.id, winner="Variant B")

        updated = manager.get_test(test.id)

        assert updated.winner == "Variant B"

    def test_generate_title_variants_with_mock_llm(self, manager):
        """Test title variant generation"""
        with patch("src.optimization.ab_testing.OllamaClient") as mock_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.generate.return_value = "Variant 1\nVariant 2\nVariant 3"
            mock_class.return_value = mock_llm

            variants = manager.generate_title_variants("Original Title", count=3)

        assert len(variants) == 3

    def test_generate_title_variants_llm_unavailable(self, manager):
        """Test title generation when LLM unavailable"""
        with patch("src.optimization.ab_testing.OllamaClient") as mock_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_class.return_value = mock_llm

            variants = manager.generate_title_variants("Original Title")

        assert variants == ["Original Title"]


class TestNormCdf:
    """Tests for normal CDF approximation"""

    def test_norm_cdf_zero(self):
        """Test CDF at zero"""
        result = _norm_cdf(0)

        assert abs(result - 0.5) < 0.001

    def test_norm_cdf_positive(self):
        """Test CDF for positive values"""
        result = _norm_cdf(2)

        # Should be close to 0.9772
        assert result > 0.97
        assert result < 0.98

    def test_norm_cdf_negative(self):
        """Test CDF for negative values"""
        result = _norm_cdf(-2)

        # Should be close to 0.0228
        assert result > 0.02
        assert result < 0.03

    def test_norm_cdf_symmetry(self):
        """Test CDF symmetry"""
        pos = _norm_cdf(1.5)
        neg = _norm_cdf(-1.5)

        assert abs(pos + neg - 1.0) < 0.001


class TestABTestManagerPersistence:
    """Tests for database persistence"""

    def test_data_persists_across_instances(self, temp_dir):
        """Test that data persists when creating new manager instance"""
        db_path = temp_dir / "persist_test.db"

        # Create test with first instance
        manager1 = ABTestManager(db_path=db_path)
        test = manager1.create_test("vid1", "title", "A", "B")
        test_id = test.id

        # Record some data
        manager1.record_impression(test_id, "a")
        manager1.record_click(test_id, "a")

        # Create new instance and verify data persisted
        manager2 = ABTestManager(db_path=db_path)
        retrieved = manager2.get_test(test_id)

        assert retrieved is not None
        assert retrieved.variant_a.impressions == 1
        assert retrieved.variant_a.clicks == 1
