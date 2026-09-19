"""Progress tracking for resumable PDF bookmark processing."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field

from pdf_bookmarks.utils import Log


@dataclass
class ProgressState:
    """Represents the current processing progress with detailed page-level tracking."""

    input_path: str
    output_path: str
    status: str  # scanning_toc, calculating_offset, verifying_offset, generating_bookmarks, refining_bookmarks, applying_bookmarks, completed, error

    # TOC scanning
    toc_scan_complete: bool = False  # Whether TOC scanning is complete
    toc_pages_count: int = 0
    toc_scan_current_page: int = 0  # Which PDF page we're scanning for TOC
    content_start_index: int = 0  # 0-based index of first non-TOC page
    toc_start_index: int = -1  # 0-based index of first TOC page, -1 means not set

    # Offset calculation
    offset_calculated: bool = False
    page_offset: int = 0
    first_entry_title: str = ""
    first_entry_toc_page: int = 0
    first_entry_actual_page: int = 0
    offset_search_start_page: int = 0  # Where we started searching
    offset_search_end_page: int = 0    # Where we ended searching
    offset_search_current_page: int = 0  # Current page being checked

    # Offset verification
    verification_passed: bool = False
    verification_entries: List[Dict[str, Any]] = field(default_factory=list)  # List of verification entries
    verification_current: int = 0  # Which verification we're on (1 or 2)

    # Bookmark generation - detailed per TOC page
    toc_pages: List = field(default_factory=list)  # Cached TOC page images
    toc_page_processed: List[bool] = field(default_factory=list)  # Track which TOC pages are done
    toc_page_results: Dict[str, str] = field(default_factory=dict)
    current_toc_page_index: int = 0
    accumulated_bookmarks: str = ""
    last_entry: str = ""
    total_bookmarks_generated: int = 0

    # Bookmark refinement
    refined_bookmarks: str = ""

    # Error handling
    error_message: str = ""
    error_step: str = ""
    error_page_context: str = ""  # Which page we were processing when error occurred
    can_retry: bool = True

    # Metadata
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressState":
        """Create from dictionary."""
        return cls(**data)

    def has_error(self) -> bool:
        """Check if this state has an error."""
        return self.status == "error"

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        if self.status == "scanning_toc":
            return f"Scanning for TOC pages (checked page {self.toc_scan_current_page})"
        elif self.status == "calculating_offset":
            if self.offset_search_current_page > 0:
                return f"Calculating offset (searching page {self.offset_search_current_page}/{self.offset_search_end_page})"
            return "Calculating offset..."
        elif self.status == "verifying_offset":
            return f"Verifying offset ({self.verification_current}/2 confirmations)"
        elif self.status == "generating_bookmarks":
            done = sum(self.toc_page_processed) if self.toc_page_processed else 0
            return f"Generating bookmarks (TOC page {done}/{self.toc_pages_count})"
        elif self.status == "refining_bookmarks":
            return "Refining bookmarks..."
        elif self.status == "applying_bookmarks":
            return "Applying bookmarks to PDF..."
        elif self.status == "error":
            return f"Error at {self.error_step}: {self.error_message}"
        return self.status

    def get_previous_step(self) -> str:
        """Get the step before the error for retry."""
        if self.error_step == "calculating_offset":
            return "calculating_offset"
        elif self.error_step == "verifying_offset":
            return "calculating_offset"  # Go back to offset calculation
        elif self.error_step == "generating_bookmarks":
            # Restart from the current TOC page
            return "generating_bookmarks"
        elif self.error_step == "refining_bookmarks":
            return "refining_bookmarks"
        elif self.error_step == "applying_bookmarks":
            return "refining_bookmarks"
        return self.error_step


class ProgressManager:
    """Manages progress tracking for resumable processing."""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.progress_file = self._get_progress_file()

    def _get_progress_file(self) -> str:
        """Get the progress file path for the input."""
        input_path = Path(self.input_path)
        return str(input_path.parent / f"{input_path.name}.tmp.json")

    def save(self, state: ProgressState) -> None:
        """Save progress state to file."""
        state.timestamp = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self) -> Optional[ProgressState]:
        """Load progress state from file."""
        if not os.path.exists(self.progress_file):
            return None

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProgressState.from_dict(data)
        except Exception:
            return None

    def delete(self) -> None:
        """Delete progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def exists(self) -> bool:
        """Check if progress file exists."""
        return os.path.exists(self.progress_file)

    def can_resume(self, input_path: str, output_path: str) -> bool:
        """Check if existing progress can be resumed."""
        state = self.load()
        if not state:
            return False

        # Verify input file matches
        if state.input_path != input_path:
            return False

        # Can resume if not completed
        return state.status != "completed"

    def print_resume_info(self) -> None:
        """Print detailed resume information to user."""
        state = self.load()
        if not state:
            return

        Log.separator()
        Log.info(f"Found previous progress from {state.timestamp}")
        Log.detail(f"Status: {state.get_progress_summary()}")

        if state.has_error():
            Log.error(f"Previous error: {state.error_message}")
            if state.error_page_context:
                Log.detail(f"Error context: {state.error_page_context}")
            Log.detail(f"Error occurred at: {state.error_step}")
        elif state.status == "calculating_offset":
            Log.detail(f"First entry: '{state.first_entry_title}' (TOC page {state.first_entry_toc_page})")
            if state.offset_search_current_page > 0:
                Log.detail(f"Searched up to page {state.offset_search_current_page}")
        elif state.status == "verifying_offset":
            Log.detail(f"Verification: {state.verification_current}/2 passed")
            if state.verification_entries:
                for i, entry in enumerate(state.verification_entries, 1):
                    status = "✓" if entry.get("passed") else "✗"
                    Log.detail(f"  {status} Verification {i}: '{entry.get('title', 'N/A')}' at page {entry.get('page', 'N/A')}")
        elif state.status == "generating_bookmarks":
            done = sum(state.toc_page_processed) if state.toc_page_processed else 0
            Log.detail(f"Processed {done}/{state.toc_pages_count} TOC pages")
            Log.detail(f"Generated {state.total_bookmarks_generated} bookmarks so far")
            if state.current_toc_page_index < len(state.toc_page_processed):
                Log.detail(f"Next: TOC page {state.current_toc_page_index + 1}")
        elif state.status == "refining_bookmarks":
            Log.detail(f"Generated {state.total_bookmarks_generated} bookmarks, now refining...")
        elif self.status == "applying_bookmarks":
            Log.detail("Applying bookmarks to PDF...")

        Log.separator()

    def mark_error(self, state: ProgressState, error_message: str, error_step: str,
                   page_context: str = "", can_retry: bool = True) -> None:
        """Mark the state as having an error."""
        state.status = "error"
        state.error_message = error_message
        state.error_step = error_step
        state.error_page_context = page_context
        state.can_retry = can_retry
        self.save(state)
