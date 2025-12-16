"""Unit tests for leaderboard parser module."""

from __future__ import annotations

import pandas as pd
import pytest

from admet.leaderboard.parser import (
    extract_value_uncertainty,
    find_user_rank,
    normalize_user_cell,
    to_dataframe,
)


class TestToDataframe:
    """Tests for to_dataframe function."""

    def test_empty_none(self) -> None:
        """Test with None input."""
        result = to_dataframe(None)
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_already_dataframe(self) -> None:
        """Test with DataFrame input returns same DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_dataframe(df)
        assert result is df

    def test_dict_with_headers(self) -> None:
        """Test dict format with 'headers' key."""
        data = {"headers": ["rank", "user"], "data": [[1, "alice"], [2, "bob"]]}
        result = to_dataframe(data)
        assert list(result.columns) == ["rank", "user"]
        assert len(result) == 2
        assert result.iloc[0]["user"] == "alice"

    def test_dict_with_columns(self) -> None:
        """Test dict format with 'columns' key."""
        data = {"columns": ["rank", "user"], "data": [[1, "alice"], [2, "bob"]]}
        result = to_dataframe(data)
        assert list(result.columns) == ["rank", "user"]
        assert len(result) == 2

    def test_dict_without_headers(self) -> None:
        """Test dict format without headers/columns."""
        data = {"data": [[1, 2], [3, 4]]}
        result = to_dataframe(data)
        assert list(result.columns) == ["col0", "col1"]
        assert len(result) == 2

    def test_dict_empty_data(self) -> None:
        """Test dict with empty data."""
        data = {"headers": ["rank", "user"], "data": []}
        result = to_dataframe(data)
        assert result.empty
        assert list(result.columns) == ["rank", "user"]

    def test_list_of_dicts(self) -> None:
        """Test list of dictionaries."""
        data = [{"rank": 1, "user": "alice"}, {"rank": 2, "user": "bob"}]
        result = to_dataframe(data)
        assert "rank" in result.columns
        assert "user" in result.columns
        assert len(result) == 2

    def test_list_of_lists(self) -> None:
        """Test list of lists."""
        data = [[1, "alice"], [2, "bob"]]
        result = to_dataframe(data)
        assert list(result.columns) == ["col0", "col1"]
        assert len(result) == 2

    def test_empty_list(self) -> None:
        """Test empty list."""
        result = to_dataframe([])
        assert result.empty


class TestNormalizeUserCell:
    """Tests for normalize_user_cell function."""

    def test_normal_string(self) -> None:
        """Test with normal string."""
        assert normalize_user_cell("Alice") == "alice"

    def test_whitespace(self) -> None:
        """Test with whitespace."""
        assert normalize_user_cell("  Alice  ") == "alice"

    def test_none_input(self) -> None:
        """Test with None."""
        assert normalize_user_cell(None) == ""

    def test_number(self) -> None:
        """Test with number."""
        assert normalize_user_cell(123) == "123"


class TestExtractValueUncertainty:
    """Tests for extract_value_uncertainty function."""

    def test_plus_minus_format(self) -> None:
        """Test '0.35 +/- 0.01' format."""
        val, err = extract_value_uncertainty("0.35 +/- 0.01")
        assert val == pytest.approx(0.35)
        assert err == pytest.approx(0.01)

    def test_pm_symbol_format(self) -> None:
        """Test '0.42 ± 0.02' format."""
        val, err = extract_value_uncertainty("0.42 ± 0.02")
        assert val == pytest.approx(0.42)
        assert err == pytest.approx(0.02)

    def test_plain_float(self) -> None:
        """Test plain float string."""
        val, err = extract_value_uncertainty("0.35")
        assert val == pytest.approx(0.35)
        assert err is None

    def test_negative_value(self) -> None:
        """Test negative value."""
        val, err = extract_value_uncertainty("-0.35 +/- 0.01")
        assert val == pytest.approx(-0.35)
        assert err == pytest.approx(0.01)

    def test_none_input(self) -> None:
        """Test with None."""
        val, err = extract_value_uncertainty(None)
        assert val is None
        assert err is None

    def test_invalid_string(self) -> None:
        """Test with invalid string."""
        val, err = extract_value_uncertainty("invalid")
        assert val is None
        assert err is None

    def test_whitespace_tolerance(self) -> None:
        """Test with extra whitespace."""
        val, err = extract_value_uncertainty("  0.35  +/-  0.01  ")
        assert val == pytest.approx(0.35)
        assert err == pytest.approx(0.01)


class TestFindUserRank:
    """Tests for find_user_rank function."""

    def test_simple_match(self) -> None:
        """Test simple username match."""
        df = pd.DataFrame({"user": ["alice", "bob", "charlie"]})
        rank = find_user_rank(df, "bob")
        assert rank == 2

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        df = pd.DataFrame({"user": ["alice", "bob", "charlie"]})
        rank = find_user_rank(df, "BOB")
        assert rank == 2

    def test_markdown_link(self) -> None:
        """Test matching markdown link format."""
        df = pd.DataFrame({"user": ["alice", "[bob](https://example.com)", "charlie"]})
        rank = find_user_rank(df, "bob")
        assert rank == 2

    def test_user_not_found(self) -> None:
        """Test when user not found."""
        df = pd.DataFrame({"user": ["alice", "bob", "charlie"]})
        rank = find_user_rank(df, "dave")
        assert rank is None

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame({"user": []})
        rank = find_user_rank(df, "alice")
        assert rank is None

    def test_no_user_column(self) -> None:
        """Test with DataFrame missing user column."""
        df = pd.DataFrame({"rank": [1, 2, 3]})
        rank = find_user_rank(df, "alice")
        assert rank is None

    def test_user_column_variations(self) -> None:
        """Test with different user column names."""
        df = pd.DataFrame({"User": ["alice", "bob"]})
        rank = find_user_rank(df, "alice")
        assert rank == 1

    def test_first_match(self) -> None:
        """Test returns first match."""
        df = pd.DataFrame({"user": ["alice", "bob", "alice"]})
        rank = find_user_rank(df, "alice")
        assert rank == 1
