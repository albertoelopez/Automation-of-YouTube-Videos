"""Analytics tracking for video generation and performance"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Iterator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class GenerationLog:
    """Log entry for a video generation"""
    id: int | None = None
    timestamp: datetime = None
    topic: str = ""
    template: str | None = None
    style: str = ""
    duration_target: int = 0
    duration_actual: float = 0.0

    # Performance metrics
    generation_time: float = 0.0  # seconds
    script_time: float = 0.0
    tts_time: float = 0.0
    video_time: float = 0.0

    # Output info
    video_path: str = ""
    file_size: int = 0  # bytes

    # Upload status
    youtube_uploaded: bool = False
    youtube_id: str = ""
    tiktok_uploaded: bool = False
    tiktok_id: str = ""

    # Errors
    success: bool = True
    error: str = ""


@dataclass
class VideoStats:
    """Performance statistics for uploaded videos"""
    video_id: str
    platform: str  # youtube, tiktok
    title: str = ""

    # Engagement metrics
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0

    # Performance
    watch_time_hours: float = 0.0
    avg_view_duration: float = 0.0
    ctr: float = 0.0  # click-through rate

    # Timestamps
    published_at: datetime = None
    last_updated: datetime = None


class AnalyticsTracker:
    """
    Track and analyze video generation and performance.

    Features:
    - Log all generation attempts
    - Track upload status
    - Fetch YouTube/TikTok analytics
    - Generate reports
    """

    def __init__(self, db_path: Path | str = "data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    topic TEXT,
                    template TEXT,
                    style TEXT,
                    duration_target INTEGER,
                    duration_actual REAL,
                    generation_time REAL,
                    script_time REAL,
                    tts_time REAL,
                    video_time REAL,
                    video_path TEXT,
                    file_size INTEGER,
                    youtube_uploaded INTEGER DEFAULT 0,
                    youtube_id TEXT,
                    tiktok_uploaded INTEGER DEFAULT 0,
                    tiktok_id TEXT,
                    success INTEGER DEFAULT 1,
                    error TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT,
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    watch_time_hours REAL DEFAULT 0,
                    avg_view_duration REAL DEFAULT 0,
                    ctr REAL DEFAULT 0,
                    published_at TEXT,
                    last_updated TEXT,
                    UNIQUE(video_id, platform)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp
                ON generation_logs(timestamp)
            """)

    def log_generation(self, log: GenerationLog) -> int:
        """Log a video generation"""
        if log.timestamp is None:
            log.timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO generation_logs (
                    timestamp, topic, template, style,
                    duration_target, duration_actual,
                    generation_time, script_time, tts_time, video_time,
                    video_path, file_size,
                    youtube_uploaded, youtube_id,
                    tiktok_uploaded, tiktok_id,
                    success, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.timestamp.isoformat(),
                log.topic, log.template, log.style,
                log.duration_target, log.duration_actual,
                log.generation_time, log.script_time, log.tts_time, log.video_time,
                log.video_path, log.file_size,
                int(log.youtube_uploaded), log.youtube_id,
                int(log.tiktok_uploaded), log.tiktok_id,
                int(log.success), log.error,
            ))
            return cursor.lastrowid

    def update_upload_status(
        self,
        log_id: int,
        platform: str,
        video_id: str,
    ):
        """Update upload status for a generation log"""
        with sqlite3.connect(self.db_path) as conn:
            if platform == "youtube":
                conn.execute(
                    "UPDATE generation_logs SET youtube_uploaded=1, youtube_id=? WHERE id=?",
                    (video_id, log_id)
                )
            elif platform == "tiktok":
                conn.execute(
                    "UPDATE generation_logs SET tiktok_uploaded=1, tiktok_id=? WHERE id=?",
                    (video_id, log_id)
                )

    def get_logs(
        self,
        days: int = 30,
        limit: int = 100,
    ) -> list[GenerationLog]:
        """Get generation logs"""
        since = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM generation_logs
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (since, limit))

            logs = []
            for row in cursor:
                log = GenerationLog(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    topic=row["topic"],
                    template=row["template"],
                    style=row["style"],
                    duration_target=row["duration_target"],
                    duration_actual=row["duration_actual"],
                    generation_time=row["generation_time"],
                    script_time=row["script_time"],
                    tts_time=row["tts_time"],
                    video_time=row["video_time"],
                    video_path=row["video_path"],
                    file_size=row["file_size"],
                    youtube_uploaded=bool(row["youtube_uploaded"]),
                    youtube_id=row["youtube_id"] or "",
                    tiktok_uploaded=bool(row["tiktok_uploaded"]),
                    tiktok_id=row["tiktok_id"] or "",
                    success=bool(row["success"]),
                    error=row["error"] or "",
                )
                logs.append(log)

            return logs

    def get_summary(self, days: int = 30) -> dict:
        """Get summary statistics"""
        since = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            total = conn.execute(
                "SELECT COUNT(*) FROM generation_logs WHERE timestamp >= ?",
                (since,)
            ).fetchone()[0]

            successful = conn.execute(
                "SELECT COUNT(*) FROM generation_logs WHERE timestamp >= ? AND success=1",
                (since,)
            ).fetchone()[0]

            youtube_uploads = conn.execute(
                "SELECT COUNT(*) FROM generation_logs WHERE timestamp >= ? AND youtube_uploaded=1",
                (since,)
            ).fetchone()[0]

            tiktok_uploads = conn.execute(
                "SELECT COUNT(*) FROM generation_logs WHERE timestamp >= ? AND tiktok_uploaded=1",
                (since,)
            ).fetchone()[0]

            # Averages
            avg_gen_time = conn.execute(
                "SELECT AVG(generation_time) FROM generation_logs WHERE timestamp >= ? AND success=1",
                (since,)
            ).fetchone()[0] or 0

            avg_duration = conn.execute(
                "SELECT AVG(duration_actual) FROM generation_logs WHERE timestamp >= ? AND success=1",
                (since,)
            ).fetchone()[0] or 0

            total_file_size = conn.execute(
                "SELECT SUM(file_size) FROM generation_logs WHERE timestamp >= ? AND success=1",
                (since,)
            ).fetchone()[0] or 0

            # Popular templates
            templates = conn.execute("""
                SELECT template, COUNT(*) as count
                FROM generation_logs
                WHERE timestamp >= ? AND template IS NOT NULL
                GROUP BY template
                ORDER BY count DESC
                LIMIT 5
            """, (since,)).fetchall()

            return {
                "period_days": days,
                "total_generated": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "youtube_uploads": youtube_uploads,
                "tiktok_uploads": tiktok_uploads,
                "avg_generation_time": avg_gen_time,
                "avg_video_duration": avg_duration,
                "total_storage_mb": total_file_size / (1024 * 1024),
                "popular_templates": dict(templates),
            }

    def update_video_stats(self, stats: VideoStats):
        """Update or insert video statistics"""
        if stats.last_updated is None:
            stats.last_updated = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO video_stats (
                    video_id, platform, title,
                    views, likes, comments, shares,
                    watch_time_hours, avg_view_duration, ctr,
                    published_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, platform) DO UPDATE SET
                    views=excluded.views,
                    likes=excluded.likes,
                    comments=excluded.comments,
                    shares=excluded.shares,
                    watch_time_hours=excluded.watch_time_hours,
                    avg_view_duration=excluded.avg_view_duration,
                    ctr=excluded.ctr,
                    last_updated=excluded.last_updated
            """, (
                stats.video_id, stats.platform, stats.title,
                stats.views, stats.likes, stats.comments, stats.shares,
                stats.watch_time_hours, stats.avg_view_duration, stats.ctr,
                stats.published_at.isoformat() if stats.published_at else None,
                stats.last_updated.isoformat(),
            ))

    def get_video_stats(
        self,
        platform: str | None = None,
        limit: int = 50,
    ) -> list[VideoStats]:
        """Get video statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if platform:
                cursor = conn.execute(
                    "SELECT * FROM video_stats WHERE platform=? ORDER BY views DESC LIMIT ?",
                    (platform, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM video_stats ORDER BY views DESC LIMIT ?",
                    (limit,)
                )

            stats = []
            for row in cursor:
                s = VideoStats(
                    video_id=row["video_id"],
                    platform=row["platform"],
                    title=row["title"] or "",
                    views=row["views"],
                    likes=row["likes"],
                    comments=row["comments"],
                    shares=row["shares"],
                    watch_time_hours=row["watch_time_hours"],
                    avg_view_duration=row["avg_view_duration"],
                    ctr=row["ctr"],
                )
                if row["published_at"]:
                    s.published_at = datetime.fromisoformat(row["published_at"])
                if row["last_updated"]:
                    s.last_updated = datetime.fromisoformat(row["last_updated"])
                stats.append(s)

            return stats

    def print_summary(self, days: int = 30):
        """Print analytics summary"""
        summary = self.get_summary(days)

        console.print(Panel(
            f"[bold]Analytics Summary (Last {days} Days)[/bold]",
            style="cyan",
        ))

        # Generation stats table
        gen_table = Table(title="Generation Statistics")
        gen_table.add_column("Metric", style="cyan")
        gen_table.add_column("Value", style="bold")

        gen_table.add_row("Total Generated", str(summary["total_generated"]))
        gen_table.add_row("Successful", f"[green]{summary['successful']}[/green]")
        gen_table.add_row("Failed", f"[red]{summary['failed']}[/red]")
        gen_table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        gen_table.add_row("YouTube Uploads", str(summary["youtube_uploads"]))
        gen_table.add_row("TikTok Uploads", str(summary["tiktok_uploads"]))
        gen_table.add_row("Avg Generation Time", f"{summary['avg_generation_time']:.1f}s")
        gen_table.add_row("Avg Video Duration", f"{summary['avg_video_duration']:.1f}s")
        gen_table.add_row("Total Storage Used", f"{summary['total_storage_mb']:.1f} MB")

        console.print(gen_table)

        # Popular templates
        if summary["popular_templates"]:
            console.print("\n[bold]Popular Templates:[/bold]")
            for template, count in summary["popular_templates"].items():
                console.print(f"  • {template}: {count} videos")

    def print_recent_logs(self, limit: int = 10):
        """Print recent generation logs"""
        logs = self.get_logs(limit=limit)

        table = Table(title="Recent Generations")
        table.add_column("Time", style="dim")
        table.add_column("Topic")
        table.add_column("Duration")
        table.add_column("Gen Time")
        table.add_column("Status")
        table.add_column("Uploads")

        for log in logs:
            status = "[green]✓[/green]" if log.success else f"[red]✗ {log.error[:20]}[/red]"

            uploads = []
            if log.youtube_uploaded:
                uploads.append("YT")
            if log.tiktok_uploaded:
                uploads.append("TT")

            table.add_row(
                log.timestamp.strftime("%m/%d %H:%M"),
                log.topic[:25] + "..." if len(log.topic) > 25 else log.topic,
                f"{log.duration_actual:.0f}s",
                f"{log.generation_time:.0f}s",
                status,
                ", ".join(uploads) or "-",
            )

        console.print(table)

    def print_video_performance(self, limit: int = 10):
        """Print video performance stats"""
        stats = self.get_video_stats(limit=limit)

        if not stats:
            console.print("[dim]No video stats available yet[/dim]")
            return

        table = Table(title="Video Performance")
        table.add_column("Platform")
        table.add_column("Title")
        table.add_column("Views", justify="right")
        table.add_column("Likes", justify="right")
        table.add_column("Comments", justify="right")
        table.add_column("CTR", justify="right")

        for s in stats:
            table.add_row(
                s.platform.upper(),
                s.title[:30] + "..." if len(s.title) > 30 else s.title,
                f"{s.views:,}",
                f"{s.likes:,}",
                f"{s.comments:,}",
                f"{s.ctr:.1f}%",
            )

        console.print(table)
