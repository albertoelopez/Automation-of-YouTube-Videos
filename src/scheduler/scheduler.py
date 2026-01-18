"""Video generation and upload scheduler"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Callable, Literal
from enum import Enum
from rich.console import Console
from rich.table import Table

console = Console()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """A scheduled video generation/upload job"""
    id: str
    topic: str
    scheduled_time: datetime
    status: JobStatus = JobStatus.PENDING

    # Generation options
    template: str | None = None
    style: str = "informative"
    duration: int = 30

    # Upload options
    upload_youtube: bool = False
    upload_tiktok: bool = False
    privacy: str = "private"

    # Results
    video_path: str | None = None
    youtube_url: str | None = None
    tiktok_url: str | None = None
    error: str | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["scheduled_time"] = self.scheduled_time.isoformat()
        data["status"] = self.status.value
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledJob":
        """Create from dictionary"""
        data["scheduled_time"] = datetime.fromisoformat(data["scheduled_time"])
        data["status"] = JobStatus(data["status"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)


class Scheduler:
    """
    Job scheduler for automated video generation and upload.

    Features:
    - Schedule jobs for specific times
    - Recurring schedules (daily, weekly)
    - Persistent job storage
    - Background execution
    """

    def __init__(
        self,
        jobs_file: Path | str = "config/scheduled_jobs.json",
        check_interval: int = 60,  # seconds
    ):
        self.jobs_file = Path(jobs_file)
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self.check_interval = check_interval

        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable] = []

        self._load_jobs()

    def _load_jobs(self):
        """Load jobs from file"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file) as f:
                    data = json.load(f)
                for job_data in data.get("jobs", []):
                    job = ScheduledJob.from_dict(job_data)
                    self._jobs[job.id] = job
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load jobs: {e}[/yellow]")

    def _save_jobs(self):
        """Save jobs to file"""
        data = {
            "jobs": [job.to_dict() for job in self._jobs.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.jobs_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique job ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def add_job(
        self,
        topic: str,
        scheduled_time: datetime | str,
        **kwargs,
    ) -> ScheduledJob:
        """Add a new scheduled job"""
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time)

        job = ScheduledJob(
            id=self._generate_id(),
            topic=topic,
            scheduled_time=scheduled_time,
            **kwargs,
        )

        self._jobs[job.id] = job
        self._save_jobs()

        console.print(f"[green]Job scheduled: {job.id} at {scheduled_time}[/green]")
        return job

    def add_recurring(
        self,
        topic: str,
        start_time: datetime,
        interval: Literal["daily", "weekly", "hourly"],
        count: int = 7,
        **kwargs,
    ) -> list[ScheduledJob]:
        """Add multiple recurring jobs"""
        intervals = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
        }

        delta = intervals[interval]
        jobs = []

        for i in range(count):
            scheduled_time = start_time + (delta * i)
            job = self.add_job(topic, scheduled_time, **kwargs)
            jobs.append(job)

        console.print(f"[green]Created {len(jobs)} recurring jobs ({interval})[/green]")
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                self._save_jobs()
                console.print(f"[yellow]Job cancelled: {job_id}[/yellow]")
                return True
        return False

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Get a job by ID"""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[ScheduledJob]:
        """List jobs, optionally filtered by status"""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by scheduled time
        jobs.sort(key=lambda j: j.scheduled_time)

        return jobs[:limit]

    def get_pending_jobs(self) -> list[ScheduledJob]:
        """Get all pending jobs that are due"""
        now = datetime.now()
        return [
            job for job in self._jobs.values()
            if job.status == JobStatus.PENDING and job.scheduled_time <= now
        ]

    def _execute_job(self, job: ScheduledJob):
        """Execute a single job"""
        console.print(f"\n[cyan]Executing job: {job.id} - {job.topic}[/cyan]")

        job.status = JobStatus.RUNNING
        self._save_jobs()

        try:
            # Import here to avoid circular imports
            from ..pipeline import Pipeline
            from ..utils.config import load_config

            config = load_config()
            pipeline = Pipeline(config)

            # Generate video
            result = pipeline.generate(
                topic=job.topic,
                style=job.style,
                duration=job.duration,
            )

            if result:
                job.video_path = str(result.video_path)

                # Upload to YouTube if requested
                if job.upload_youtube:
                    try:
                        from ..upload.youtube import YouTubeUploader
                        uploader = YouTubeUploader()
                        upload_result = uploader.upload_from_result(
                            video_path=result.video_path,
                            metadata=result.metadata,
                            privacy_status=job.privacy,
                        )
                        if upload_result:
                            job.youtube_url = upload_result.url
                    except Exception as e:
                        console.print(f"[yellow]YouTube upload failed: {e}[/yellow]")

                # Upload to TikTok if requested
                if job.upload_tiktok:
                    try:
                        from ..upload.tiktok import TikTokUploader
                        uploader = TikTokUploader()
                        upload_result = uploader.upload(
                            video_path=result.video_path,
                            description=result.metadata.get("description", ""),
                            tags=result.metadata.get("hashtags", []),
                        )
                        if upload_result:
                            job.tiktok_url = upload_result.share_url
                    except Exception as e:
                        console.print(f"[yellow]TikTok upload failed: {e}[/yellow]")

                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                console.print(f"[green]Job completed: {job.id}[/green]")

            else:
                job.status = JobStatus.FAILED
                job.error = "Video generation returned None"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            console.print(f"[red]Job failed: {job.id} - {e}[/red]")

        self._save_jobs()

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(job)
            except Exception:
                pass

    def run_once(self):
        """Check and execute due jobs once"""
        pending = self.get_pending_jobs()

        if pending:
            console.print(f"[cyan]Found {len(pending)} pending jobs[/cyan]")

        for job in pending:
            self._execute_job(job)

    def start(self):
        """Start the scheduler in background thread"""
        if self._running:
            console.print("[yellow]Scheduler already running[/yellow]")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        console.print("[green]Scheduler started[/green]")

    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        console.print("[yellow]Scheduler stopped[/yellow]")

    def _run_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                console.print(f"[red]Scheduler error: {e}[/red]")

            time.sleep(self.check_interval)

    def on_job_complete(self, callback: Callable[[ScheduledJob], None]):
        """Register callback for job completion"""
        self._callbacks.append(callback)

    def print_status(self):
        """Print scheduler status"""
        table = Table(title="Scheduled Jobs")
        table.add_column("ID", style="cyan")
        table.add_column("Topic")
        table.add_column("Scheduled")
        table.add_column("Status")
        table.add_column("Uploads")

        for job in self.list_jobs(limit=20):
            status_color = {
                JobStatus.PENDING: "yellow",
                JobStatus.RUNNING: "blue",
                JobStatus.COMPLETED: "green",
                JobStatus.FAILED: "red",
                JobStatus.CANCELLED: "dim",
            }.get(job.status, "white")

            uploads = []
            if job.upload_youtube:
                uploads.append("YT")
            if job.upload_tiktok:
                uploads.append("TT")

            table.add_row(
                job.id,
                job.topic[:30] + "..." if len(job.topic) > 30 else job.topic,
                job.scheduled_time.strftime("%Y-%m-%d %H:%M"),
                f"[{status_color}]{job.status.value}[/{status_color}]",
                ", ".join(uploads) or "-",
            )

        console.print(table)

        # Stats
        pending = len([j for j in self._jobs.values() if j.status == JobStatus.PENDING])
        completed = len([j for j in self._jobs.values() if j.status == JobStatus.COMPLETED])
        failed = len([j for j in self._jobs.values() if j.status == JobStatus.FAILED])

        console.print(f"\n[dim]Pending: {pending} | Completed: {completed} | Failed: {failed}[/dim]")


def run_scheduler(
    jobs_file: Path | str = "config/scheduled_jobs.json",
    foreground: bool = True,
):
    """Run the scheduler"""
    scheduler = Scheduler(jobs_file=jobs_file)

    if foreground:
        console.print("[cyan]Running scheduler in foreground (Ctrl+C to stop)[/cyan]")
        scheduler.print_status()

        try:
            while True:
                scheduler.run_once()
                time.sleep(scheduler.check_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped[/yellow]")
    else:
        scheduler.start()
        return scheduler
