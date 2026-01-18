"""A/B Testing for thumbnails, titles, and descriptions"""
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import random
import math

console = Console()


@dataclass
class Variant:
    """A test variant (A or B)"""
    id: str
    name: str
    value: str  # Title text, thumbnail path, or description
    impressions: int = 0
    clicks: int = 0
    watch_time: float = 0.0  # Average watch time in seconds
    likes: int = 0
    comments: int = 0
    shares: int = 0

    @property
    def ctr(self) -> float:
        """Click-through rate"""
        if self.impressions == 0:
            return 0.0
        return (self.clicks / self.impressions) * 100

    @property
    def engagement_rate(self) -> float:
        """Engagement rate (likes + comments + shares) / impressions"""
        if self.impressions == 0:
            return 0.0
        return ((self.likes + self.comments + self.shares) / self.impressions) * 100


@dataclass
class ABTest:
    """An A/B test configuration"""
    id: str
    video_id: str
    test_type: Literal["title", "thumbnail", "description"]
    variant_a: Variant
    variant_b: Variant
    status: Literal["active", "completed", "paused"] = "active"
    winner: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    min_impressions: int = 1000  # Minimum impressions before declaring winner
    confidence_level: float = 0.95  # Statistical significance threshold

    def get_winner(self) -> tuple[Variant | None, float]:
        """
        Determine winner using statistical significance.
        Returns (winner_variant, confidence) or (None, 0) if inconclusive.
        """
        if self.variant_a.impressions < self.min_impressions:
            return None, 0.0
        if self.variant_b.impressions < self.min_impressions:
            return None, 0.0

        # Calculate Z-score for CTR difference
        p_a = self.variant_a.ctr / 100
        p_b = self.variant_b.ctr / 100
        n_a = self.variant_a.impressions
        n_b = self.variant_b.impressions

        # Pooled proportion
        p_pool = (self.variant_a.clicks + self.variant_b.clicks) / (n_a + n_b)

        if p_pool == 0 or p_pool == 1:
            return None, 0.0

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

        if se == 0:
            return None, 0.0

        # Z-score
        z = (p_a - p_b) / se

        # Two-tailed p-value approximation
        confidence = 1 - 2 * (1 - _norm_cdf(abs(z)))

        if confidence >= self.confidence_level:
            if p_a > p_b:
                return self.variant_a, confidence
            else:
                return self.variant_b, confidence

        return None, confidence


def _norm_cdf(x: float) -> float:
    """Approximate normal CDF"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class ABTestManager:
    """
    Manage A/B tests for video optimization.

    Features:
    - Create tests for titles, thumbnails, descriptions
    - Track impressions and engagement
    - Statistical significance calculation
    - Automatic winner selection
    - Generate variations using AI
    """

    def __init__(self, db_path: Path | str = "data/ab_tests.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tests (
                    id TEXT PRIMARY KEY,
                    video_id TEXT,
                    test_type TEXT,
                    status TEXT DEFAULT 'active',
                    winner TEXT,
                    created_at TEXT,
                    ended_at TEXT,
                    min_impressions INTEGER DEFAULT 1000,
                    confidence_level REAL DEFAULT 0.95
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS variants (
                    id TEXT PRIMARY KEY,
                    test_id TEXT,
                    name TEXT,
                    value TEXT,
                    impressions INTEGER DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    watch_time REAL DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    FOREIGN KEY (test_id) REFERENCES tests(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_variants_test ON variants(test_id)
            """)

    def create_test(
        self,
        video_id: str,
        test_type: Literal["title", "thumbnail", "description"],
        variant_a_value: str,
        variant_b_value: str,
        min_impressions: int = 1000,
    ) -> ABTest:
        """Create a new A/B test"""
        import uuid

        test_id = str(uuid.uuid4())[:8]
        variant_a = Variant(
            id=f"{test_id}_a",
            name="Variant A",
            value=variant_a_value,
        )
        variant_b = Variant(
            id=f"{test_id}_b",
            name="Variant B",
            value=variant_b_value,
        )

        test = ABTest(
            id=test_id,
            video_id=video_id,
            test_type=test_type,
            variant_a=variant_a,
            variant_b=variant_b,
            min_impressions=min_impressions,
        )

        self._save_test(test)
        return test

    def _save_test(self, test: ABTest):
        """Save test to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tests
                (id, video_id, test_type, status, winner, created_at, ended_at, min_impressions, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test.id,
                test.video_id,
                test.test_type,
                test.status,
                test.winner,
                test.created_at.isoformat(),
                test.ended_at.isoformat() if test.ended_at else None,
                test.min_impressions,
                test.confidence_level,
            ))

            for variant in [test.variant_a, test.variant_b]:
                conn.execute("""
                    INSERT OR REPLACE INTO variants
                    (id, test_id, name, value, impressions, clicks, watch_time, likes, comments, shares)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    variant.id,
                    test.id,
                    variant.name,
                    variant.value,
                    variant.impressions,
                    variant.clicks,
                    variant.watch_time,
                    variant.likes,
                    variant.comments,
                    variant.shares,
                ))

    def get_test(self, test_id: str) -> ABTest | None:
        """Get a test by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tests WHERE id = ?", (test_id,)
            ).fetchone()

            if not row:
                return None

            variants = conn.execute(
                "SELECT * FROM variants WHERE test_id = ?", (test_id,)
            ).fetchall()

            variant_a = None
            variant_b = None

            for v in variants:
                variant = Variant(
                    id=v["id"],
                    name=v["name"],
                    value=v["value"],
                    impressions=v["impressions"],
                    clicks=v["clicks"],
                    watch_time=v["watch_time"],
                    likes=v["likes"],
                    comments=v["comments"],
                    shares=v["shares"],
                )
                if v["id"].endswith("_a"):
                    variant_a = variant
                else:
                    variant_b = variant

            if not variant_a or not variant_b:
                return None

            return ABTest(
                id=row["id"],
                video_id=row["video_id"],
                test_type=row["test_type"],
                variant_a=variant_a,
                variant_b=variant_b,
                status=row["status"],
                winner=row["winner"],
                created_at=datetime.fromisoformat(row["created_at"]),
                ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                min_impressions=row["min_impressions"],
                confidence_level=row["confidence_level"],
            )

    def list_tests(
        self,
        status: str | None = None,
        video_id: str | None = None,
    ) -> list[ABTest]:
        """List all tests"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT id FROM tests WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if video_id:
                query += " AND video_id = ?"
                params.append(video_id)

            query += " ORDER BY created_at DESC"

            rows = conn.execute(query, params).fetchall()
            return [self.get_test(row[0]) for row in rows if self.get_test(row[0])]

    def record_impression(self, test_id: str, variant: Literal["a", "b"]):
        """Record an impression for a variant"""
        variant_id = f"{test_id}_{variant}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE variants SET impressions = impressions + 1 WHERE id = ?",
                (variant_id,)
            )

    def record_click(self, test_id: str, variant: Literal["a", "b"]):
        """Record a click for a variant"""
        variant_id = f"{test_id}_{variant}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE variants SET clicks = clicks + 1 WHERE id = ?",
                (variant_id,)
            )

    def record_engagement(
        self,
        test_id: str,
        variant: Literal["a", "b"],
        likes: int = 0,
        comments: int = 0,
        shares: int = 0,
        watch_time: float = 0,
    ):
        """Record engagement metrics"""
        variant_id = f"{test_id}_{variant}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE variants SET
                    likes = likes + ?,
                    comments = comments + ?,
                    shares = shares + ?,
                    watch_time = (watch_time * impressions + ?) / (impressions + 1)
                WHERE id = ?
            """, (likes, comments, shares, watch_time, variant_id))

    def select_variant(self, test_id: str) -> Literal["a", "b"]:
        """
        Select which variant to show using epsilon-greedy strategy.
        Returns "a" or "b".
        """
        test = self.get_test(test_id)
        if not test or test.status != "active":
            return "a"

        # Exploration vs exploitation (10% exploration)
        epsilon = 0.1

        if random.random() < epsilon:
            # Explore: random selection
            return random.choice(["a", "b"])
        else:
            # Exploit: choose better performer
            if test.variant_a.ctr >= test.variant_b.ctr:
                return "a"
            return "b"

    def check_winner(self, test_id: str) -> tuple[bool, str | None, float]:
        """
        Check if a test has a statistically significant winner.
        Returns (has_winner, winner_name, confidence)
        """
        test = self.get_test(test_id)
        if not test:
            return False, None, 0.0

        winner, confidence = test.get_winner()

        if winner:
            return True, winner.name, confidence
        return False, None, confidence

    def end_test(self, test_id: str, winner: str | None = None):
        """End a test and optionally set winner"""
        test = self.get_test(test_id)
        if not test:
            return

        test.status = "completed"
        test.ended_at = datetime.now()

        if winner is None:
            # Auto-determine winner
            w, _ = test.get_winner()
            if w:
                test.winner = w.name
        else:
            test.winner = winner

        self._save_test(test)

    def generate_title_variants(
        self,
        original_title: str,
        count: int = 3,
    ) -> list[str]:
        """Generate title variations using AI"""
        from ..llm.ollama import OllamaClient

        llm = OllamaClient()
        if not llm.is_available():
            return [original_title]

        prompt = f"""Generate {count} alternative titles for this YouTube video title:
"{original_title}"

Requirements:
- Keep similar meaning but vary the style
- Try different hooks: question, number, curiosity gap
- Keep under 60 characters
- Make them click-worthy but not clickbait

Output only the titles, one per line, no numbering or bullets."""

        try:
            response = llm.generate(prompt, temperature=0.9)
            variants = [
                line.strip().strip('"').strip("'")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return variants[:count]
        except Exception:
            return [original_title]

    def generate_description_variants(
        self,
        original_description: str,
        count: int = 2,
    ) -> list[str]:
        """Generate description variations using AI"""
        from ..llm.ollama import OllamaClient

        llm = OllamaClient()
        if not llm.is_available():
            return [original_description]

        prompt = f"""Generate {count} alternative descriptions for this YouTube video:

Original:
{original_description}

Requirements:
- Keep the core message
- Vary the hook and structure
- Include call-to-action
- Optimize for engagement

Separate each description with "---"
Output only the descriptions."""

        try:
            response = llm.generate(prompt, temperature=0.8)
            variants = [
                desc.strip()
                for desc in response.split("---")
                if desc.strip() and len(desc.strip()) > 20
            ]
            return variants[:count]
        except Exception:
            return [original_description]

    def print_test(self, test: ABTest):
        """Print test details"""
        winner, confidence = test.get_winner()

        content = f"""[bold]Test ID:[/bold] {test.id}
[bold]Video:[/bold] {test.video_id}
[bold]Type:[/bold] {test.test_type}
[bold]Status:[/bold] {test.status}
[bold]Created:[/bold] {test.created_at.strftime('%Y-%m-%d %H:%M')}

[cyan]Variant A:[/cyan]
  Value: {test.variant_a.value[:50]}{'...' if len(test.variant_a.value) > 50 else ''}
  Impressions: {test.variant_a.impressions:,}
  Clicks: {test.variant_a.clicks:,}
  CTR: {test.variant_a.ctr:.2f}%
  Engagement: {test.variant_a.engagement_rate:.2f}%

[cyan]Variant B:[/cyan]
  Value: {test.variant_b.value[:50]}{'...' if len(test.variant_b.value) > 50 else ''}
  Impressions: {test.variant_b.impressions:,}
  Clicks: {test.variant_b.clicks:,}
  CTR: {test.variant_b.ctr:.2f}%
  Engagement: {test.variant_b.engagement_rate:.2f}%

[bold]Statistical Confidence:[/bold] {confidence*100:.1f}%"""

        if winner:
            content += f"\n[green]Winner: {winner.name}[/green]"
        elif test.variant_a.impressions < test.min_impressions:
            remaining = test.min_impressions - test.variant_a.impressions
            content += f"\n[yellow]Need {remaining:,} more impressions[/yellow]"

        console.print(Panel(content, title=f"A/B Test: {test.test_type.title()}"))

    def print_all_tests(self):
        """Print summary of all tests"""
        tests = self.list_tests()

        if not tests:
            console.print("[dim]No A/B tests found[/dim]")
            return

        table = Table(title="A/B Tests")
        table.add_column("ID")
        table.add_column("Type")
        table.add_column("Video")
        table.add_column("Status")
        table.add_column("A CTR")
        table.add_column("B CTR")
        table.add_column("Winner")

        for test in tests:
            winner, _ = test.get_winner()
            winner_str = winner.name if winner else "-"

            table.add_row(
                test.id,
                test.test_type,
                test.video_id[:15] + "..." if len(test.video_id) > 15 else test.video_id,
                test.status,
                f"{test.variant_a.ctr:.1f}%",
                f"{test.variant_b.ctr:.1f}%",
                winner_str,
            )

        console.print(table)
