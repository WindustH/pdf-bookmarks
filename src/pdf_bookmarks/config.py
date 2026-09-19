"""Configuration management."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(str(PROJECT_ROOT / "model.env"))


@dataclass
class Config:
    """Configuration for PDF bookmark processing."""

    api_key: str = os.getenv("API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "")
    vision_model: str = os.getenv("VISION_MODEL", "")
    text_model: str = os.getenv("TEXT_MODEL", "")
    refine_timeout: float = float(os.getenv("REFINE_TIMEOUT", "600"))
    toc_workers: int = int(os.getenv("TOC_WORKERS", "4"))
    temp_bookmark_file: str = "bookmarks.txt"
