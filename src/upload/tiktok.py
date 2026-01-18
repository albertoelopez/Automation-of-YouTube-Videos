"""TikTok upload functionality"""
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from rich.console import Console

console = Console()


@dataclass
class TikTokUploadResult:
    """Result of a TikTok upload"""
    video_id: str
    share_url: str
    status: str


class TikTokUploader:
    """
    TikTok video uploader.

    Methods:
    1. Browser automation (Selenium/Playwright) - works but against ToS
    2. TikTok API (requires business account approval)
    3. Third-party services (Repurpose.io, etc.)

    This implementation uses browser automation as a reference.
    For production, consider using official API or approved services.
    """

    def __init__(
        self,
        cookies_path: Path | str = "config/tiktok_cookies.json",
        headless: bool = True,
    ):
        self.cookies_path = Path(cookies_path)
        self.headless = headless
        self._browser = None
        self._page = None

    def is_configured(self) -> bool:
        """Check if TikTok cookies exist"""
        return self.cookies_path.exists()

    def _check_playwright(self) -> bool:
        """Check if Playwright is installed"""
        try:
            from playwright.sync_api import sync_playwright
            return True
        except ImportError:
            return False

    def _init_browser(self):
        """Initialize Playwright browser"""
        if self._browser is not None:
            return

        if not self._check_playwright():
            raise RuntimeError(
                "Playwright not installed. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )

        # Load cookies if available
        if self.cookies_path.exists():
            with open(self.cookies_path) as f:
                cookies = json.load(f)
            self._context.add_cookies(cookies)

        self._page = self._context.new_page()

    def save_cookies(self):
        """Save current browser cookies"""
        if self._context:
            cookies = self._context.cookies()
            with open(self.cookies_path, "w") as f:
                json.dump(cookies, f)
            console.print(f"[green]Cookies saved to {self.cookies_path}[/green]")

    def login_manual(self):
        """Open browser for manual login"""
        self._init_browser()

        # Use non-headless for manual login
        if self.headless:
            self._browser.close()
            self._playwright.stop()
            self._browser = None

            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self._context = self._browser.new_context()
            self._page = self._context.new_page()

        console.print("[cyan]Opening TikTok login page...[/cyan]")
        console.print("[yellow]Please log in manually, then press Enter when done.[/yellow]")

        self._page.goto("https://www.tiktok.com/login")
        input("Press Enter after logging in...")

        self.save_cookies()
        console.print("[green]Login successful! Cookies saved.[/green]")

    def upload(
        self,
        video_path: Path | str,
        description: str = "",
        tags: list[str] | None = None,
        privacy: Literal["public", "friends", "private"] = "public",
        allow_comments: bool = True,
        allow_duet: bool = True,
        allow_stitch: bool = True,
    ) -> TikTokUploadResult | None:
        """
        Upload a video to TikTok.

        Note: This uses browser automation which may break with TikTok updates.
        For production use, consider the official TikTok API.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            console.print(f"[red]Video file not found: {video_path}[/red]")
            return None

        if not self.is_configured():
            console.print("[red]TikTok not configured. Run login first.[/red]")
            return None

        try:
            self._init_browser()

            console.print("[cyan]Navigating to TikTok upload page...[/cyan]")
            self._page.goto("https://www.tiktok.com/upload")
            time.sleep(3)

            # Check if logged in
            if "login" in self._page.url.lower():
                console.print("[red]Not logged in. Please run login first.[/red]")
                return None

            # Wait for upload button
            console.print("[cyan]Uploading video...[/cyan]")

            # Find file input and upload
            file_input = self._page.locator('input[type="file"]').first
            file_input.set_input_files(str(video_path))

            # Wait for upload to complete
            console.print("[cyan]Waiting for upload to process...[/cyan]")
            time.sleep(10)  # TikTok processing time varies

            # Add description with hashtags
            full_description = description
            if tags:
                hashtags = " ".join(f"#{tag}" for tag in tags)
                full_description = f"{description} {hashtags}"

            # Find caption input
            caption_input = self._page.locator('[data-e2e="caption-textarea"]').first
            if caption_input:
                caption_input.fill(full_description[:2200])  # TikTok limit

            # Set privacy (if not public)
            if privacy != "public":
                # Click privacy dropdown
                privacy_btn = self._page.locator('[data-e2e="privacy-select"]').first
                if privacy_btn:
                    privacy_btn.click()
                    time.sleep(1)
                    # Select option
                    self._page.locator(f'text="{privacy.title()}"').click()

            # Click post button
            console.print("[cyan]Posting video...[/cyan]")
            post_btn = self._page.locator('[data-e2e="post-button"]').first
            if post_btn:
                post_btn.click()
                time.sleep(5)

            # Get result
            console.print("[green]Upload complete![/green]")

            return TikTokUploadResult(
                video_id="pending",  # TikTok doesn't immediately provide this
                share_url="https://www.tiktok.com/@username",  # Would need to extract
                status="uploaded",
            )

        except Exception as e:
            console.print(f"[red]Upload failed: {e}[/red]")
            return None

    def close(self):
        """Close browser"""
        if self._browser:
            self._browser.close()
        if hasattr(self, "_playwright") and self._playwright:
            self._playwright.stop()


class TikTokAPIUploader:
    """
    TikTok API uploader (for approved business accounts).

    Requires:
    - TikTok for Developers account
    - Approved app with Video Upload permission
    - OAuth tokens

    See: https://developers.tiktok.com/doc/upload-video-api
    """

    def __init__(
        self,
        client_key: str,
        client_secret: str,
        access_token: str | None = None,
    ):
        self.client_key = client_key
        self.client_secret = client_secret
        self.access_token = access_token

    def upload(
        self,
        video_path: Path | str,
        **kwargs,
    ) -> TikTokUploadResult | None:
        """Upload using TikTok API"""
        # This would implement the official API
        # https://developers.tiktok.com/doc/upload-video-api

        console.print("[yellow]TikTok API upload not yet implemented.[/yellow]")
        console.print("For now, use browser-based upload or third-party services.")
        return None
