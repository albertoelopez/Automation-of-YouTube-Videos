"""Upload module for YouTube and TikTok"""
from .youtube import YouTubeUploader, upload_video
from .tiktok import TikTokUploader

__all__ = ["YouTubeUploader", "upload_video", "TikTokUploader"]
