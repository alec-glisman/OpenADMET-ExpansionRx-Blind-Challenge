"""Configuration for leaderboard scraping and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LeaderboardConfig:
    """Configuration for leaderboard scraping and analysis.

    Attributes
    ----------
    space : str
        HuggingFace Space identifier (format: "owner/space-name")
    target_user : str
        Username to analyze in leaderboards (case-insensitive)
    cache_dir : Path
        Directory for caching leaderboard data and outputs
    endpoints : List[str]
        List of task endpoint names to analyze
    """

    space: str = "openadmet/OpenADMET-ExpansionRx-Challenge"
    target_user: str = "aglisman"
    cache_dir: Path = field(default_factory=lambda: Path("assets/submissions"))
    endpoints: List[str] = field(
        default_factory=lambda: [
            "LogD",
            "KSOL",
            "MLM CLint",
            "HLM CLint",
            "Caco-2 Permeability Efflux",
            "Caco-2 Permeability Papp A>B",
            "MPPB",
            "MBPB",
            "MGMB",
        ]
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.space:
            raise ValueError("space must be non-empty")
        if not self.target_user:
            raise ValueError("target_user must be non-empty")

        # Ensure cache_dir is Path
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def all_endpoints(self) -> List[str]:
        """Return all endpoints including 'Average' for overall rankings.

        Returns
        -------
        List[str]
            ['Average'] + endpoints list
        """
        return ["Average"] + self.endpoints

    def get_output_dir(self, timestamp: str | None = None) -> Path:
        """Get timestamped output directory for results.

        Parameters
        ----------
        timestamp : str, optional
            Timestamp string (e.g., "2025-12-16"). If None, no subdirectory.

        Returns
        -------
        Path
            Output directory path
        """
        if timestamp:
            return self.cache_dir / timestamp
        return self.cache_dir
