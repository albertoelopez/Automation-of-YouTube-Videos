"""Unit tests for thumbnail generator"""
import pytest
from pathlib import Path
from PIL import Image

from src.images.thumbnail import (
    ThumbnailGenerator,
    ThumbnailStyle,
    THUMBNAIL_STYLES,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
)


class TestThumbnailStyle:
    """Tests for ThumbnailStyle dataclass"""

    def test_default_style(self):
        """Test default style values"""
        style = ThumbnailStyle(name="test")

        assert style.name == "test"
        assert style.background_color == (20, 20, 20)
        assert style.text_color == (255, 255, 255)
        assert style.font_size == 120
        assert style.text_position == "center"

    def test_custom_style(self):
        """Test custom style values"""
        style = ThumbnailStyle(
            name="custom",
            background_color=(255, 0, 0),
            text_color=(0, 0, 0),
            accent_color=(0, 255, 0),
            font_size=80,
            text_position="bottom",
        )

        assert style.background_color == (255, 0, 0)
        assert style.text_color == (0, 0, 0)
        assert style.text_position == "bottom"


class TestPredefinedStyles:
    """Tests for predefined thumbnail styles"""

    def test_all_styles_exist(self):
        """Test that all expected styles are defined"""
        expected_styles = ["bold", "minimal", "gaming", "news", "tutorial", "viral"]

        for style_name in expected_styles:
            assert style_name in THUMBNAIL_STYLES

    def test_styles_have_required_fields(self):
        """Test that all styles have required fields"""
        for name, style in THUMBNAIL_STYLES.items():
            assert style.name == name
            assert isinstance(style.background_color, tuple)
            assert isinstance(style.text_color, tuple)
            assert isinstance(style.font_size, int)
            assert style.text_position in ["center", "bottom", "top"]


class TestThumbnailGenerator:
    """Tests for ThumbnailGenerator class"""

    @pytest.fixture
    def generator(self, temp_dir):
        """Create generator with temp output dir"""
        return ThumbnailGenerator(output_dir=temp_dir / "thumbnails")

    def test_init_creates_output_dir(self, temp_dir):
        """Test that init creates output directory"""
        output_dir = temp_dir / "thumb_output"
        gen = ThumbnailGenerator(output_dir=output_dir)

        assert output_dir.exists()

    def test_generate_basic_thumbnail(self, generator):
        """Test generating a basic thumbnail"""
        result = generator.generate(
            text="Test Thumbnail",
            style="bold",
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"

        # Verify dimensions
        img = Image.open(result)
        assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_generate_with_all_styles(self, generator):
        """Test generating thumbnails with all predefined styles"""
        for style_name in THUMBNAIL_STYLES.keys():
            result = generator.generate(
                text=f"Test {style_name}",
                style=style_name,
            )

            assert result is not None
            assert result.exists()

    def test_generate_with_custom_style(self, generator):
        """Test generating with a custom style object"""
        custom_style = ThumbnailStyle(
            name="custom",
            background_color=(100, 100, 100),
            text_color=(255, 255, 0),
            font_size=100,
        )

        result = generator.generate(
            text="Custom Style Test",
            style=custom_style,
        )

        assert result is not None
        assert result.exists()

    def test_generate_with_background_image(self, generator, sample_image):
        """Test generating with a background image"""
        result = generator.generate(
            text="With Background",
            style="bold",
            background_image=sample_image,
        )

        assert result is not None
        assert result.exists()

    def test_generate_with_output_name(self, generator):
        """Test custom output filename"""
        result = generator.generate(
            text="Named Output",
            style="bold",
            output_name="my_thumbnail",
        )

        assert result is not None
        assert result.name == "my_thumbnail.png"

    def test_generate_variants(self, generator):
        """Test generating multiple variants"""
        variants = generator.generate_variants(
            text="A/B Test Thumbnail",
            styles=["bold", "minimal", "viral"],
        )

        assert len(variants) == 3
        for path in variants:
            assert path.exists()

    def test_generate_variants_default_styles(self, generator):
        """Test variant generation with default styles"""
        variants = generator.generate_variants(text="Default Variants")

        assert len(variants) == 3  # Default is ["bold", "minimal", "viral"]

    def test_text_wrapping(self, generator):
        """Test that long text is wrapped"""
        long_text = "This is a very long title that should be wrapped across multiple lines for readability"

        result = generator.generate(
            text=long_text,
            style="bold",
        )

        assert result is not None
        assert result.exists()

    def test_generate_from_template(self, generator):
        """Test template-based generation"""
        template = "{count} Amazing {topic} Tips"
        variables = {"count": "5", "topic": "Python"}

        result = generator.generate_from_template(
            template=template,
            variables=variables,
            style="bold",
        )

        assert result is not None
        assert result.exists()

    def test_list_styles(self):
        """Test listing available styles"""
        styles = ThumbnailGenerator.list_styles()

        assert isinstance(styles, list)
        assert len(styles) == len(THUMBNAIL_STYLES)
        for style in styles:
            assert "name" in style
            assert "text_color" in style

    def test_enhance_image(self, generator, sample_image):
        """Test image enhancement"""
        enhanced = generator.enhance_image(
            sample_image,
            brightness=1.2,
            contrast=1.3,
            saturation=1.1,
        )

        assert enhanced is not None
        assert enhanced.size[0] > 0

    def test_thumbnail_dimensions(self, generator):
        """Test that generated thumbnails have correct YouTube dimensions"""
        result = generator.generate(text="Dimensions Test", style="bold")

        img = Image.open(result)
        assert img.size == (1280, 720)

    def test_thumbnail_is_rgb(self, generator):
        """Test that generated thumbnails are RGB"""
        result = generator.generate(text="RGB Test", style="bold")

        img = Image.open(result)
        assert img.mode == "RGB"


class TestThumbnailGeneratorEdgeCases:
    """Edge case tests for thumbnail generator"""

    @pytest.fixture
    def generator(self, temp_dir):
        """Create generator with temp output dir"""
        return ThumbnailGenerator(output_dir=temp_dir / "thumbnails")

    def test_empty_text(self, generator):
        """Test with empty text"""
        result = generator.generate(text="", style="bold")

        # Should still generate a thumbnail (with no text)
        assert result is not None
        assert result.exists()

    def test_special_characters(self, generator):
        """Test text with special characters"""
        result = generator.generate(
            text="Test! @#$% 🎉 Special <chars>",
            style="bold",
        )

        assert result is not None
        assert result.exists()

    def test_unicode_text(self, generator):
        """Test text with unicode characters"""
        result = generator.generate(
            text="日本語テスト 中文测试",
            style="bold",
        )

        assert result is not None
        assert result.exists()

    def test_very_long_text_truncation(self, generator):
        """Test that very long text doesn't break layout"""
        very_long = "A" * 500

        result = generator.generate(text=very_long, style="bold")

        assert result is not None
        assert result.exists()

        # Verify image is still valid
        img = Image.open(result)
        assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_invalid_style_falls_back(self, generator):
        """Test that invalid style name falls back to bold"""
        result = generator.generate(
            text="Invalid Style",
            style="nonexistent_style",
        )

        assert result is not None
        assert result.exists()

    def test_background_image_resize(self, generator, temp_dir):
        """Test that background images are properly resized"""
        # Create a small image
        small_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        small_path = temp_dir / "small.png"
        small_img.save(small_path)

        result = generator.generate(
            text="Small BG Test",
            style="bold",
            background_image=small_path,
        )

        assert result is not None
        output_img = Image.open(result)
        assert output_img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_background_image_crop(self, generator, temp_dir):
        """Test that background images with wrong aspect ratio are cropped"""
        # Create a wide image
        wide_img = Image.new("RGB", (3000, 500), color=(0, 255, 0))
        wide_path = temp_dir / "wide.png"
        wide_img.save(wide_path)

        result = generator.generate(
            text="Wide BG Test",
            style="bold",
            background_image=wide_path,
        )

        assert result is not None
        output_img = Image.open(result)
        assert output_img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
