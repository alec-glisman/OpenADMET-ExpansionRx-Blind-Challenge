"""Unit tests for leaderboard client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from admet.leaderboard.client import LeaderboardClient
from admet.leaderboard.config import LeaderboardConfig


class TestLeaderboardClient:
    """Tests for LeaderboardClient class."""

    @pytest.fixture
    def config(self, tmp_path) -> LeaderboardConfig:
        """Create test configuration."""
        return LeaderboardConfig(
            space="test/space",
            target_user="testuser",
            cache_dir=tmp_path / "cache",
        )

    @pytest.fixture
    def client(self, config) -> LeaderboardClient:
        """Create test client."""
        return LeaderboardClient(config)

    def test_init(self, client, config) -> None:
        """Test client initialization."""
        assert client.config == config
        assert client.client is None

    @patch("admet.leaderboard.client.Client")
    def test_get_client(self, mock_client_class, client) -> None:
        """Test _get_client creates client."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        result = client._get_client()

        assert result is mock_instance
        assert client.client is mock_instance
        mock_client_class.assert_called_once_with("test/space")

    @patch("admet.leaderboard.client.Client")
    def test_get_client_caches(self, mock_client_class, client) -> None:
        """Test _get_client caches connection."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client._get_client()
        client._get_client()  # Second call

        # Should only create client once
        mock_client_class.assert_called_once()

    @patch("admet.leaderboard.client.Client")
    def test_get_client_connection_error(self, mock_client_class, client) -> None:
        """Test _get_client handles connection errors."""
        mock_client_class.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to connect to Gradio space"):
            client._get_client()

    def test_find_refresh_dependency(self, client) -> None:
        """Test _find_refresh_dependency finds correct fn_index."""
        # Mock client config
        mock_client = MagicMock()
        mock_client.config = {
            "dependencies": [
                {"inputs": [], "outputs": ["out1"]},  # Wrong number of outputs
                {"inputs": [], "outputs": ["out1", "out2", "out3"]},  # Correct!
                {"inputs": ["in1"], "outputs": ["out1", "out2", "out3"]},  # Has inputs
            ]
        }
        client.client = mock_client

        fn_index = client._find_refresh_dependency(n_expected_outputs=3)

        # Should find index 1 (0 inputs, 3 outputs)
        assert fn_index == 1

    def test_find_refresh_dependency_no_match(self, client) -> None:
        """Test _find_refresh_dependency raises when no match found."""
        mock_client = MagicMock()
        mock_client.config = {
            "dependencies": [
                {"inputs": [], "outputs": ["out1"]},  # Wrong number
            ]
        }
        client.client = mock_client

        with pytest.raises(RuntimeError, match="No Gradio dependency found"):
            client._find_refresh_dependency(n_expected_outputs=3)

    @patch("admet.leaderboard.client.Client")
    def test_fetch_all_tables_success(self, mock_client_class, client) -> None:
        """Test successful table fetch."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.config = {
            "dependencies": [
                {"inputs": [], "outputs": ["o1", "o2", "o3"]},
            ]
        }

        # Mock predict response
        mock_instance.predict.return_value = [
            {"headers": ["rank", "user"], "data": [[1, "alice"]]},
            {"headers": ["rank", "user"], "data": [[1, "bob"]]},
            {"headers": ["rank", "user"], "data": [[1, "charlie"]]},
        ]

        mock_client_class.return_value = mock_instance
        client.client = mock_instance

        # Fetch tables
        result = client.fetch_all_tables()

        # Verify
        assert len(result) == 3
        assert "Average" in result
        assert isinstance(result["Average"], pd.DataFrame)
        assert len(result["Average"]) == 1

    @patch("admet.leaderboard.client.Client")
    def test_fetch_all_tables_retry(self, mock_client_class, client) -> None:
        """Test fetch retries on failure."""
        mock_instance = MagicMock()
        mock_instance.config = {"dependencies": [{"inputs": [], "outputs": ["o1", "o2", "o3"]}]}

        # First call fails, second succeeds
        mock_instance.predict.side_effect = [
            Exception("Temporary failure"),
            [
                {"headers": ["rank"], "data": [[1]]},
                {"headers": ["rank"], "data": [[2]]},
                {"headers": ["rank"], "data": [[3]]},
            ],
        ]

        mock_client_class.return_value = mock_instance
        client.client = mock_instance

        result = client.fetch_all_tables(max_retries=2, retry_delay=0.1)

        assert len(result) == 3
        assert mock_instance.predict.call_count == 2

    @patch("admet.leaderboard.client.Client")
    def test_fetch_all_tables_max_retries_exceeded(self, mock_client_class, client) -> None:
        """Test fetch fails after max retries."""
        mock_instance = MagicMock()
        mock_instance.config = {"dependencies": [{"inputs": [], "outputs": ["o1", "o2", "o3"]}]}
        mock_instance.predict.side_effect = Exception("Persistent failure")

        mock_client_class.return_value = mock_instance
        client.client = mock_instance

        with pytest.raises(RuntimeError, match="Failed to fetch leaderboard"):
            client.fetch_all_tables(max_retries=2, retry_delay=0.1)

    def test_close(self, client) -> None:
        """Test close method."""
        client.client = MagicMock()
        client.close()
        assert client.client is None


@pytest.mark.integration
class TestLeaderboardClientIntegration:
    """Integration tests for real API calls (slow, marked for optional execution)."""

    @pytest.mark.skip(reason="Requires network access and is slow")
    def test_real_fetch(self) -> None:
        """Test fetching from real HuggingFace Space."""
        config = LeaderboardConfig()
        client = LeaderboardClient(config)

        tables = client.fetch_all_tables()

        assert len(tables) > 0
        assert "Average" in tables
        assert isinstance(tables["Average"], pd.DataFrame)
        assert not tables["Average"].empty
