"""
Unit tests for Bloomberg API and Excel data sources.

Tests include:
- Bloomberg API connection and data fetching
- Bloomberg Excel parsing with error handling
- Hybrid Bloomberg source with fallback logic
- Mock responses for testing without Terminal access
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd
import numpy as np

# Import Bloomberg data sources
import sys
sys.path.insert(0, '..')
from _data_sources import (
    BloombergAPIDataSource,
    BloombergExcelDataSource,
    HybridBloombergDataSource,
    BloombergAPIError,
    BloombergTerminalNotRunning,
    BloombergAuthenticationError
)


class TestBloombergAPIDataSource(unittest.TestCase):
    """Test Bloomberg Terminal API data source."""

    def setUp(self):
        """Set up test fixtures."""
        self.securities = ["LF98TRUU Index", "LUACTRUU Index"]
        self.fields = ["OAS", "PX_LAST"]
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2020, 12, 31)

    def test_initialization(self):
        """Test BloombergAPIDataSource initialization."""
        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        self.assertEqual(source.securities, self.securities)
        self.assertEqual(source.fields, self.fields)
        self.assertEqual(source.start_date, self.start_date)
        self.assertEqual(source.end_date, self.end_date)
        self.assertEqual(source.host, "localhost")
        self.assertEqual(source.port, 8194)

    def test_import_blpapi_missing(self):
        """Test handling of missing blpapi library."""
        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        with patch('builtins.__import__', side_effect=ImportError):
            with self.assertRaises(ImportError) as context:
                source._import_blpapi()

            self.assertIn("blpapi", str(context.exception))

    @patch('_data_sources.BloombergAPIDataSource._import_blpapi')
    def test_create_session_terminal_not_running(self, mock_import):
        """Test error when Bloomberg Terminal is not running."""
        # Mock blpapi module
        mock_blpapi = MagicMock()
        mock_session = MagicMock()
        mock_session.start.return_value = False
        mock_blpapi.Session.return_value = mock_session
        mock_import.return_value = mock_blpapi

        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        with self.assertRaises(BloombergTerminalNotRunning):
            source._create_session()

    @patch('_data_sources.BloombergAPIDataSource._import_blpapi')
    def test_create_session_service_failed(self, mock_import):
        """Test error when Bloomberg service cannot be opened."""
        # Mock blpapi module
        mock_blpapi = MagicMock()
        mock_session = MagicMock()
        mock_session.start.return_value = True
        mock_session.openService.return_value = False
        mock_blpapi.Session.return_value = mock_session
        mock_import.return_value = mock_blpapi

        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        with self.assertRaises(BloombergAuthenticationError):
            source._create_session()

    @patch('_data_sources.BloombergAPIDataSource._import_blpapi')
    def test_create_session_success(self, mock_import):
        """Test successful Bloomberg session creation."""
        # Mock blpapi module
        mock_blpapi = MagicMock()
        mock_session = MagicMock()
        mock_session.start.return_value = True
        mock_session.openService.return_value = True
        mock_blpapi.Session.return_value = mock_session
        mock_import.return_value = mock_blpapi

        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        session = source._create_session()
        self.assertIsNotNone(session)
        self.assertTrue(session.start.called)
        self.assertTrue(session.openService.called)

    def test_build_dataframe_empty_data(self):
        """Test error handling for empty data."""
        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        with self.assertRaises(ValueError) as context:
            source._build_dataframe({})

        self.assertIn("No data retrieved", str(context.exception))

    def test_build_dataframe_valid_data(self):
        """Test DataFrame construction from parsed data."""
        source = BloombergAPIDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Create mock data
        mock_data = {
            "LF98TRUU Index": {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "OAS": [100.0, 105.0],
                "PX_LAST": [100.5, 101.0]
            },
            "LUACTRUU Index": {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "OAS": [200.0, 205.0],
                "PX_LAST": [200.5, 201.0]
            }
        }

        df = source._build_dataframe(mock_data)

        # Verify DataFrame structure
        self.assertIn("Dates", df.columns)
        self.assertIn("LF98TRUU_Index_OAS", df.columns)
        self.assertIn("LF98TRUU_Index_PX_LAST", df.columns)
        self.assertIn("LUACTRUU_Index_OAS", df.columns)
        self.assertIn("LUACTRUU_Index_PX_LAST", df.columns)
        self.assertEqual(len(df), 2)

        # Verify data values
        self.assertEqual(df["LF98TRUU_Index_OAS"].iloc[0], 100.0)
        self.assertEqual(df["LUACTRUU_Index_PX_LAST"].iloc[1], 201.0)


class TestBloombergExcelDataSource(unittest.TestCase):
    """Test Bloomberg Excel data source with error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_excel_path = "test_bloomberg.xlsx"

    def test_initialization(self):
        """Test BloombergExcelDataSource initialization."""
        source = BloombergExcelDataSource(
            file_path=self.test_excel_path,
            handle_errors=True
        )

        self.assertEqual(source.file_path, self.test_excel_path)
        self.assertTrue(source.handle_errors)

    def test_handle_bloomberg_errors(self):
        """Test Bloomberg error value conversion to NaN."""
        source = BloombergExcelDataSource(file_path=self.test_excel_path)

        # Create DataFrame with Bloomberg errors
        df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5),
            "OAS": [100.0, "#N/A N/A", 105.0, "#VALUE!", 110.0],
            "Price": [100.5, 101.0, "#N/A Field Not Applicable", 102.0, 103.0]
        })

        df_cleaned = source._handle_bloomberg_errors(df)

        # Check that errors were converted to NaN
        self.assertTrue(pd.isna(df_cleaned["OAS"].iloc[1]))
        self.assertTrue(pd.isna(df_cleaned["OAS"].iloc[3]))
        self.assertTrue(pd.isna(df_cleaned["Price"].iloc[2]))

        # Check that valid values remain
        self.assertEqual(df_cleaned["OAS"].iloc[0], 100.0)
        self.assertEqual(df_cleaned["Price"].iloc[1], 101.0)

    def test_validate_bloomberg_schema_valid(self):
        """Test schema validation with valid Bloomberg data."""
        source = BloombergExcelDataSource(file_path=self.test_excel_path)

        df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5),
            "LF98TRUU_Index_OAS": [100.0, 105.0, 110.0, 115.0, 120.0],
            "LUACTRUU_Index_OAS": [200.0, 205.0, 210.0, 215.0, 220.0]
        })

        # Should not raise exception
        source._validate_bloomberg_schema(df)

    def test_validate_bloomberg_schema_duplicate_dates(self):
        """Test schema validation fails with duplicate dates."""
        source = BloombergExcelDataSource(file_path=self.test_excel_path)

        df = pd.DataFrame({
            "Dates": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
            "OAS": [100.0, 105.0, 110.0]
        })

        with self.assertRaises(ValueError) as context:
            source._validate_bloomberg_schema(df)

        self.assertIn("duplicate dates", str(context.exception))

    def test_validate_bloomberg_schema_no_data_columns(self):
        """Test schema validation fails with no data columns."""
        source = BloombergExcelDataSource(file_path=self.test_excel_path)

        df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5)
        })

        with self.assertRaises(ValueError) as context:
            source._validate_bloomberg_schema(df)

        self.assertIn("No data columns", str(context.exception))


class TestHybridBloombergDataSource(unittest.TestCase):
    """Test Hybrid Bloomberg source with API/Excel fallback."""

    def setUp(self):
        """Set up test fixtures."""
        self.securities = ["LF98TRUU Index"]
        self.fields = ["OAS"]
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2020, 12, 31)
        self.excel_path = "test_bloomberg.xlsx"

    @patch('_data_sources.BloombergAPIDataSource.load_data')
    def test_api_first_success(self, mock_api_load):
        """Test successful API load in API-first mode."""
        # Mock successful API load
        mock_df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5),
            "LF98TRUU_Index_OAS": [100.0, 105.0, 110.0, 115.0, 120.0]
        })
        mock_api_load.return_value = mock_df

        source = HybridBloombergDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date,
            excel_fallback_path=self.excel_path,
            prefer_api=True
        )

        df = source.load_data()
        self.assertEqual(len(df), 5)
        self.assertTrue(mock_api_load.called)

    @patch('_data_sources.BloombergExcelDataSource.load_data')
    @patch('_data_sources.BloombergAPIDataSource.load_data')
    def test_api_first_fallback_to_excel(self, mock_api_load, mock_excel_load):
        """Test fallback to Excel when API fails."""
        # Mock API failure
        mock_api_load.side_effect = BloombergTerminalNotRunning("Terminal not running")

        # Mock successful Excel load
        mock_df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5),
            "LF98TRUU_Index_OAS": [100.0, 105.0, 110.0, 115.0, 120.0]
        })
        mock_excel_load.return_value = mock_df

        source = HybridBloombergDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date,
            excel_fallback_path=self.excel_path,
            prefer_api=True
        )

        df = source.load_data()
        self.assertEqual(len(df), 5)
        self.assertTrue(mock_api_load.called)
        self.assertTrue(mock_excel_load.called)

    @patch('_data_sources.BloombergExcelDataSource.load_data')
    @patch('_data_sources.BloombergAPIDataSource.load_data')
    def test_both_sources_fail(self, mock_api_load, mock_excel_load):
        """Test error when both API and Excel fail."""
        # Mock both failures
        mock_api_load.side_effect = BloombergTerminalNotRunning("Terminal not running")
        mock_excel_load.side_effect = FileNotFoundError("Excel file not found")

        source = HybridBloombergDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date,
            excel_fallback_path=self.excel_path,
            prefer_api=True
        )

        with self.assertRaises(Exception) as context:
            source.load_data()

        self.assertIn("Both Bloomberg API and Excel fallback failed", str(context.exception))

    @patch('_data_sources.BloombergExcelDataSource.load_data')
    def test_excel_first_success(self, mock_excel_load):
        """Test successful Excel load in Excel-first mode."""
        # Mock successful Excel load
        mock_df = pd.DataFrame({
            "Dates": pd.date_range("2020-01-01", periods=5),
            "LF98TRUU_Index_OAS": [100.0, 105.0, 110.0, 115.0, 120.0]
        })
        mock_excel_load.return_value = mock_df

        source = HybridBloombergDataSource(
            securities=self.securities,
            fields=self.fields,
            start_date=self.start_date,
            end_date=self.end_date,
            excel_fallback_path=self.excel_path,
            prefer_api=False
        )

        df = source.load_data()
        self.assertEqual(len(df), 5)
        self.assertTrue(mock_excel_load.called)


class TestBloombergConfiguration(unittest.TestCase):
    """Test Bloomberg configuration management."""

    @patch('_config.Config.get')
    def test_get_bloomberg_config(self, mock_get):
        """Test getting Bloomberg configuration."""
        from _config import Config

        # Mock config values
        def mock_get_side_effect(key, default=None):
            config_map = {
                "data_source.bloomberg.securities": ["LF98TRUU Index"],
                "data_source.bloomberg.fields": ["OAS"],
                "data_source.bloomberg.start_date": "2020-01-01",
                "data_source.bloomberg.end_date": "2020-12-31",
                "data_source.bloomberg.host": "localhost",
                "data_source.bloomberg.port": 8194,
                "data_source.bloomberg.timeout": 30000,
                "data_source.bloomberg.max_retries": 3,
                "data_source.bloomberg.excel_fallback": "data/bloomberg.xlsx"
            }
            return config_map.get(key, default)

        mock_get.side_effect = mock_get_side_effect

        config = Config()
        bloomberg_config = config.get_bloomberg_config()

        self.assertEqual(bloomberg_config["securities"], ["LF98TRUU Index"])
        self.assertEqual(bloomberg_config["fields"], ["OAS"])
        self.assertEqual(bloomberg_config["host"], "localhost")
        self.assertEqual(bloomberg_config["port"], 8194)


if __name__ == '__main__':
    unittest.main()
