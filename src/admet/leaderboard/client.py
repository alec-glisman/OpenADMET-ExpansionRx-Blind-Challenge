"""Client for fetching leaderboard data from Gradio HuggingFace Spaces."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
from gradio_client import Client

from .config import LeaderboardConfig
from .parser import to_dataframe

logger = logging.getLogger(__name__)


class LeaderboardClient:
    """Client for scraping leaderboard data from Gradio-based HuggingFace Spaces.

    Parameters
    ----------
    config : LeaderboardConfig
        Configuration for space and endpoints

    Attributes
    ----------
    config : LeaderboardConfig
        Configuration object
    client : Optional[Client]
        Gradio client instance (initialized on first use)

    Examples
    --------
    >>> config = LeaderboardConfig(space="openadmet/OpenADMET-ExpansionRx-Challenge")
    >>> client = LeaderboardClient(config)
    >>> tables = client.fetch_all_tables()
    >>> tables["Average"].head()
    """

    def __init__(self, config: LeaderboardConfig) -> None:
        """Initialize client with configuration.

        Parameters
        ----------
        config : LeaderboardConfig
            Configuration for space and endpoints
        """
        self.config = config
        self.client: Optional[Client] = None

    def _get_client(self) -> Client:
        """Get or create Gradio client instance.

        Returns
        -------
        Client
            Gradio client connected to configured space

        Raises
        ------
        RuntimeError
            If client connection fails
        """
        if self.client is None:
            try:
                logger.info("Connecting to HuggingFace Space: %s", self.config.space)
                self.client = Client(self.config.space)
                logger.info("Successfully connected to %s", self.config.space)
            except Exception as e:
                logger.error("Failed to connect to space %s: %s", self.config.space, e)
                raise RuntimeError(f"Failed to connect to Gradio space: {e}") from e

        return self.client

    def _find_refresh_dependency(self, n_expected_outputs: int) -> int:
        """Find the Gradio dependency (fn_index) that outputs all leaderboard tables.

        In the OpenADMET Space, refresh_if_changed returns all tables in one call:
        [per_ep[ep] for ep in ALL_EPS], where ALL_EPS = ['Average'] + ENDPOINTS.

        Parameters
        ----------
        n_expected_outputs : int
            Expected number of output tables (typically len(endpoints) + 1)

        Returns
        -------
        int
            Function index for refresh operation

        Raises
        ------
        RuntimeError
            If suitable dependency is not found
        """
        client = self._get_client()
        config_data = client.config if hasattr(client, "config") else {}

        if not isinstance(config_data, dict):
            logger.warning("Client config is not a dict, attempting direct view")
            config_data = client.view_api(return_format="dict")

        deps = config_data.get("dependencies", [])
        if not deps:
            raise RuntimeError("No dependencies found in Gradio config")

        # Find candidates: dependencies with correct number of outputs
        candidates: List[tuple[int, int, int]] = []  # (fn_index, n_inputs, n_outputs)

        for i, dep in enumerate(deps):
            if not isinstance(dep, dict):
                continue

            outputs = dep.get("outputs", [])
            inputs = dep.get("inputs", [])

            n_out = len(outputs) if isinstance(outputs, list) else 0
            n_in = len(inputs) if isinstance(inputs, list) else 0

            if n_out == n_expected_outputs:
                candidates.append((i, n_in, n_out))
                logger.debug("Found candidate fn_index=%d with %d inputs, %d outputs", i, n_in, n_out)

        if not candidates:
            raise RuntimeError(
                f"No Gradio dependency found with {n_expected_outputs} outputs. "
                f"Found {len(deps)} total dependencies."
            )

        # Prefer the one with 0 inputs (refresh_if_changed has no inputs)
        candidates.sort(key=lambda t: (t[1] != 0, t[0]))
        fn_index = candidates[0][0]

        logger.info("Using fn_index=%d for refresh operation", fn_index)
        return fn_index

    def fetch_all_tables(self, max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, pd.DataFrame]:
        """Fetch all leaderboard tables from the Gradio Space.

        Parameters
        ----------
        max_retries : int, default=3
            Maximum number of retry attempts on failure
        retry_delay : float, default=2.0
            Delay in seconds between retries

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping endpoint names to DataFrames.
            Keys include "Average" and all configured endpoints.

        Raises
        ------
        RuntimeError
            If fetching fails after all retries

        Examples
        --------
        >>> client = LeaderboardClient(config)
        >>> tables = client.fetch_all_tables()
        >>> print(tables.keys())
        dict_keys(['Average', 'LogD', 'KSOL', ...])
        """
        client = self._get_client()
        all_endpoints = self.config.all_endpoints
        n_expected = len(all_endpoints)

        # Find the refresh function
        try:
            fn_index = self._find_refresh_dependency(n_expected)
        except RuntimeError as e:
            logger.warning("Could not find refresh dependency for expected %d outputs: %s", n_expected, e)
            # Fallback: if config has a single dependency, assume it provides all tables
            config_data = client.config if hasattr(client, "config") else {}
            deps = config_data.get("dependencies", []) if isinstance(config_data, dict) else []
            if len(deps) == 1:
                dep = deps[0]
                outputs = dep.get("outputs", []) if isinstance(dep, dict) else []
                n_out = len(outputs) if isinstance(outputs, list) else 0
                if n_out > 0:
                    logger.info("Falling back to single dependency fn_index=0 with %d outputs", n_out)
                    fn_index = 0
                    n_expected = n_out
                else:
                    logger.error("Single dependency has no outputs, cannot proceed")
                    raise
            else:
                # Re-raise original error if no reasonable fallback
                raise

        # Fetch data with retries
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Fetching leaderboard data (attempt %d/%d)", attempt, max_retries)
                result = client.predict(fn_index=fn_index)

                if not isinstance(result, (list, tuple)):
                    raise ValueError(f"Unexpected result type: {type(result)}")

                if len(result) != n_expected:
                    raise ValueError(f"Expected {n_expected} tables, got {len(result)}")

                # Convert to DataFrames
                tables = {}
                for endpoint, raw_table in zip(all_endpoints, result):
                    df = to_dataframe(raw_table)
                    tables[endpoint] = df
                    logger.info(
                        "Parsed '%s' table: %d rows × %d columns",
                        endpoint,
                        len(df),
                        len(df.columns),
                    )

                return tables

            except Exception as e:
                last_exception = e
                logger.warning("Fetch attempt %d/%d failed: %s", attempt, max_retries, e)

                if attempt < max_retries:
                    logger.info("Retrying in %.1f seconds...", retry_delay)
                    time.sleep(retry_delay)

        raise RuntimeError(f"Failed to fetch leaderboard after {max_retries} attempts") from last_exception

    def close(self) -> None:
        """Close the Gradio client connection."""
        if self.client is not None:
            logger.debug("Closing Gradio client")
            self.client = None
