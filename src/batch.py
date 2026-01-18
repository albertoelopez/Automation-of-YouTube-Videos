"""Batch video generation"""
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Iterator
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .pipeline import Pipeline, GeneratedVideo
from .utils.config import Config, load_config

console = Console()


@dataclass
class BatchJob:
    """A single job in a batch"""
    topic: str
    style: str = "informative"
    duration: int = 30
    output_name: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing"""
    job: BatchJob
    success: bool
    video: GeneratedVideo | None = None
    error: str | None = None


class BatchProcessor:
    """Process multiple video generation jobs"""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.pipeline = Pipeline(self.config)

    def load_jobs_from_file(self, path: Path | str) -> list[BatchJob]:
        """
        Load batch jobs from a file.
        Supports: JSON, CSV, or plain text (one topic per line)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_json(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        else:
            return self._load_text(path)

    def _load_json(self, path: Path) -> list[BatchJob]:
        """Load jobs from JSON file"""
        with open(path) as f:
            data = json.load(f)

        jobs = []
        items = data if isinstance(data, list) else data.get("jobs", [])

        for item in items:
            if isinstance(item, str):
                jobs.append(BatchJob(topic=item))
            else:
                jobs.append(BatchJob(
                    topic=item["topic"],
                    style=item.get("style", "informative"),
                    duration=item.get("duration", 30),
                    output_name=item.get("output_name"),
                    tags=item.get("tags", []),
                    extra=item.get("extra", {}),
                ))

        return jobs

    def _load_csv(self, path: Path) -> list[BatchJob]:
        """Load jobs from CSV file"""
        jobs = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                jobs.append(BatchJob(
                    topic=row.get("topic", row.get("title", "")),
                    style=row.get("style", "informative"),
                    duration=int(row.get("duration", 30)),
                    output_name=row.get("output_name"),
                    tags=row.get("tags", "").split(",") if row.get("tags") else [],
                ))

        return jobs

    def _load_text(self, path: Path) -> list[BatchJob]:
        """Load jobs from plain text file (one topic per line)"""
        jobs = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    jobs.append(BatchJob(topic=line))

        return jobs

    def process(
        self,
        jobs: list[BatchJob],
        continue_on_error: bool = True,
        **kwargs,
    ) -> Iterator[BatchResult]:
        """
        Process batch jobs, yielding results as they complete.

        Args:
            jobs: List of batch jobs
            continue_on_error: Continue processing if a job fails
            **kwargs: Additional arguments passed to pipeline.generate()

        Yields:
            BatchResult for each completed job
        """
        console.print(f"\n[bold]Processing {len(jobs)} jobs...[/bold]\n")

        for i, job in enumerate(jobs):
            console.print(f"\n[bold cyan]═══ Job {i + 1}/{len(jobs)}: {job.topic} ═══[/bold cyan]")

            try:
                video = self.pipeline.generate(
                    topic=job.topic,
                    style=job.style,
                    duration=job.duration,
                    output_name=job.output_name,
                    **kwargs,
                )

                if video:
                    yield BatchResult(job=job, success=True, video=video)
                else:
                    yield BatchResult(job=job, success=False, error="Generation returned None")

            except Exception as e:
                error_msg = str(e)
                console.print(f"[red]Error: {error_msg}[/red]")

                yield BatchResult(job=job, success=False, error=error_msg)

                if not continue_on_error:
                    console.print("[yellow]Stopping batch due to error[/yellow]")
                    break

    def process_all(
        self,
        jobs: list[BatchJob],
        **kwargs,
    ) -> list[BatchResult]:
        """Process all jobs and return results list"""
        return list(self.process(jobs, **kwargs))

    def process_file(
        self,
        path: Path | str,
        **kwargs,
    ) -> list[BatchResult]:
        """Load jobs from file and process"""
        jobs = self.load_jobs_from_file(path)
        return self.process_all(jobs, **kwargs)

    def print_summary(self, results: list[BatchResult]) -> None:
        """Print summary of batch results"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        console.print("\n" + "═" * 50)
        console.print("[bold]Batch Processing Summary[/bold]")
        console.print("═" * 50)

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total Jobs", str(len(results)))
        table.add_row("Successful", f"[green]{len(successful)}[/green]")
        table.add_row("Failed", f"[red]{len(failed)}[/red]" if failed else "0")

        if successful:
            total_duration = sum(r.video.duration for r in successful if r.video)
            table.add_row("Total Duration", f"{total_duration:.1f}s")

        console.print(table)

        if successful:
            console.print("\n[bold green]Successful:[/bold green]")
            for r in successful:
                if r.video:
                    console.print(f"  ✓ {r.job.topic} → {r.video.video_path.name}")

        if failed:
            console.print("\n[bold red]Failed:[/bold red]")
            for r in failed:
                console.print(f"  ✗ {r.job.topic}: {r.error}")


def create_sample_batch_file(path: Path | str = "batch_jobs.json") -> Path:
    """Create a sample batch jobs file"""
    path = Path(path)

    sample = {
        "jobs": [
            {
                "topic": "5 Amazing Facts About the Ocean",
                "style": "educational",
                "duration": 30,
            },
            {
                "topic": "How to Stay Productive Working From Home",
                "style": "informative",
                "duration": 45,
            },
            {
                "topic": "The History of Coffee in 60 Seconds",
                "style": "entertaining",
                "duration": 60,
            },
        ]
    }

    with open(path, "w") as f:
        json.dump(sample, f, indent=2)

    console.print(f"[green]Sample batch file created: {path}[/green]")
    return path
