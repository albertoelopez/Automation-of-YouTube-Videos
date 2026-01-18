"""YouTube upload functionality using YouTube Data API v3"""
import json
import pickle
import httplib2
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from rich.console import Console

console = Console()

# Scopes required for uploading
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


@dataclass
class UploadResult:
    """Result of a YouTube upload"""
    video_id: str
    url: str
    title: str
    status: str


class YouTubeUploader:
    """
    YouTube video uploader using OAuth2.

    Setup:
    1. Go to https://console.cloud.google.com/
    2. Create a project
    3. Enable YouTube Data API v3
    4. Create OAuth 2.0 credentials (Desktop app)
    5. Download client_secrets.json
    6. Place in config/client_secrets.json
    """

    def __init__(
        self,
        client_secrets_path: Path | str = "config/client_secrets.json",
        credentials_path: Path | str = "config/youtube_credentials.pickle",
    ):
        self.client_secrets_path = Path(client_secrets_path)
        self.credentials_path = Path(credentials_path)
        self._youtube = None

    def is_configured(self) -> bool:
        """Check if client secrets file exists"""
        return self.client_secrets_path.exists()

    def is_authenticated(self) -> bool:
        """Check if we have valid credentials"""
        if not self.credentials_path.exists():
            return False

        try:
            with open(self.credentials_path, "rb") as f:
                credentials = pickle.load(f)
            return credentials and not credentials.invalid
        except Exception:
            return False

    def authenticate(self) -> bool:
        """
        Authenticate with YouTube.
        Opens browser for OAuth flow on first run.
        """
        if not self.is_configured():
            console.print("[red]Error: client_secrets.json not found[/red]")
            console.print("\nTo set up YouTube upload:")
            console.print("1. Go to https://console.cloud.google.com/")
            console.print("2. Create a project and enable YouTube Data API v3")
            console.print("3. Create OAuth 2.0 credentials (Desktop app)")
            console.print(f"4. Download and save as {self.client_secrets_path}")
            return False

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            credentials = None

            # Load existing credentials
            if self.credentials_path.exists():
                with open(self.credentials_path, "rb") as f:
                    credentials = pickle.load(f)

            # Refresh or get new credentials
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.client_secrets_path),
                        scopes=[YOUTUBE_UPLOAD_SCOPE],
                    )
                    credentials = flow.run_local_server(port=0)

                # Save credentials
                with open(self.credentials_path, "wb") as f:
                    pickle.dump(credentials, f)

            # Build YouTube API client
            self._youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                credentials=credentials,
            )

            console.print("[green]YouTube authentication successful![/green]")
            return True

        except ImportError:
            console.print("[red]Missing dependencies. Install with:[/red]")
            console.print("  pip install google-auth-oauthlib google-api-python-client")
            return False
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return False

    def upload(
        self,
        video_path: Path | str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        category_id: str = "22",  # 22 = People & Blogs
        privacy_status: Literal["public", "private", "unlisted"] = "private",
        made_for_kids: bool = False,
    ) -> UploadResult | None:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to video file
            title: Video title (max 100 chars)
            description: Video description
            tags: List of tags
            category_id: YouTube category ID
            privacy_status: public, private, or unlisted
            made_for_kids: Whether video is made for kids

        Returns:
            UploadResult with video ID and URL
        """
        video_path = Path(video_path)

        if not video_path.exists():
            console.print(f"[red]Video file not found: {video_path}[/red]")
            return None

        if not self._youtube:
            if not self.authenticate():
                return None

        try:
            from googleapiclient.http import MediaFileUpload

            # Prepare metadata
            body = {
                "snippet": {
                    "title": title[:100],  # Max 100 chars
                    "description": description,
                    "tags": tags or [],
                    "categoryId": category_id,
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": made_for_kids,
                },
            }

            # Upload
            console.print(f"[cyan]Uploading: {video_path.name}[/cyan]")

            media = MediaFileUpload(
                str(video_path),
                chunksize=1024 * 1024,  # 1MB chunks
                resumable=True,
            )

            request = self._youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    percent = int(status.progress() * 100)
                    console.print(f"\r[cyan]Upload progress: {percent}%[/cyan]", end="")

            console.print()

            video_id = response["id"]
            url = f"https://youtube.com/watch?v={video_id}"

            console.print(f"[green]Upload complete![/green]")
            console.print(f"[bold]URL: {url}[/bold]")

            return UploadResult(
                video_id=video_id,
                url=url,
                title=title,
                status=privacy_status,
            )

        except Exception as e:
            console.print(f"[red]Upload failed: {e}[/red]")
            return None

    def upload_from_result(
        self,
        video_path: Path | str,
        metadata: dict,
        privacy_status: Literal["public", "private", "unlisted"] = "private",
    ) -> UploadResult | None:
        """
        Upload using metadata from video generation result.

        Args:
            video_path: Path to video file
            metadata: Dict with title, description, hashtags from generation
            privacy_status: public, private, or unlisted
        """
        title = metadata.get("title", "Untitled Video")
        description = metadata.get("description", "")
        hashtags = metadata.get("hashtags", [])

        # Add hashtags to description
        if hashtags:
            hashtag_str = " ".join(f"#{tag}" for tag in hashtags)
            description = f"{description}\n\n{hashtag_str}"

        return self.upload(
            video_path=video_path,
            title=title,
            description=description,
            tags=hashtags,
            privacy_status=privacy_status,
        )


def upload_video(
    video_path: Path | str,
    title: str,
    description: str = "",
    **kwargs,
) -> UploadResult | None:
    """Convenience function to upload a video"""
    uploader = YouTubeUploader()
    return uploader.upload(video_path, title, description, **kwargs)
