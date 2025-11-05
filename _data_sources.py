################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
"""
Data source abstraction layer for loading economic data from multiple sources.

This module provides a unified interface for loading data from Excel files,
Bloomberg API, or other data sources. It allows easy switching between data
sources without changing the preprocessing pipeline.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import logging
import pathlib

import pandas as pd

logger = logging.getLogger("_data_sources")
logger.setLevel(logging.INFO)
path = pathlib.Path(__file__).parent.absolute()
handler = logging.FileHandler(path / "logs" / "_data_sources.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

################################################################################
# Abstract Base Class
################################################################################


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    All data sources must implement the load_data() method which returns
    a pandas DataFrame with a standardized schema.

    Required columns in returned DataFrame:
        - Dates: datetime column with observation dates
        - Additional columns: numeric data for economic indicators
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the source and return as a DataFrame.

        Returns:
            pd.DataFrame: Data with 'Dates' column and numeric indicator columns

        Raises:
            Exception: If data cannot be loaded
        """
        pass

    def validate_schema(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
        """
        Validate that the DataFrame has the required schema.

        Args:
            df: DataFrame to validate
            required_columns: List of column names that must be present

        Raises:
            ValueError: If required columns are missing
        """
        if "Dates" not in df.columns:
            raise ValueError("DataFrame must contain a 'Dates' column")

        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Schema validated successfully. Columns: {list(df.columns)}")


################################################################################
# Excel Data Source
################################################################################


class ExcelDataSource(DataSource):
    """
    Load data from Excel files (Bloomberg export format).

    Supports .xlsx and .xls files. Assumes first row contains column names
    and 'Dates' column contains dates in a parseable format.

    Args:
        file_path: Path to Excel file or file-like object
        sheet_name: Name or index of sheet to read (default: 0 = first sheet)
        date_column: Name of the date column (default: "Dates")

    Example:
        >>> source = ExcelDataSource("data/Economic_Data_2020_08_01.xlsx")
        >>> df = source.load_data()
        >>> print(df.head())
    """

    def __init__(
        self,
        file_path,
        sheet_name: int = 0,
        date_column: str = "Dates"
    ):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.date_column = date_column
        logger.info(f"ExcelDataSource initialized with file: {file_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file.

        Returns:
            pd.DataFrame: Data with dates and economic indicators

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read or schema is invalid
        """
        try:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            logger.info(f"Loaded {len(df)} rows from Excel file")

            # Ensure Dates column exists
            if self.date_column not in df.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in Excel")

            # Standardize column name to 'Dates'
            if self.date_column != "Dates":
                df.rename(columns={self.date_column: "Dates"}, inplace=True)

            # Convert to datetime
            df["Dates"] = pd.to_datetime(df["Dates"])

            # Validate schema
            self.validate_schema(df)

            return df

        except FileNotFoundError as e:
            logger.error(f"Excel file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Excel file: {str(e)}")
            raise ValueError(f"Failed to load Excel file: {str(e)}")


################################################################################
# Bloomberg API Data Source
################################################################################


class BloombergAPIDataSource(DataSource):
    """
    Load data from Bloomberg Terminal API (requires blpapi library).

    This is a template implementation. To use Bloomberg API:
    1. Install: pip install blpapi
    2. Configure Bloomberg Terminal connection
    3. Uncomment and adapt the implementation below

    Args:
        securities: List of Bloomberg security tickers (e.g., ["LF98TRUU Index"])
        fields: List of Bloomberg fields to retrieve (e.g., ["PX_LAST", "VOLUME"])
        start_date: Start date for historical data
        end_date: End date for historical data
        host: Bloomberg API host (default: "localhost")
        port: Bloomberg API port (default: 8194)

    Example:
        >>> source = BloombergAPIDataSource(
        ...     securities=["LF98TRUU Index", "LUACTRUU Index"],
        ...     fields=["OAS", "PX_LAST"],
        ...     start_date=datetime(2020, 1, 1),
        ...     end_date=datetime(2020, 12, 31)
        ... )
        >>> df = source.load_data()
    """

    def __init__(
        self,
        securities: List[str],
        fields: List[str],
        start_date: datetime,
        end_date: datetime,
        host: str = "localhost",
        port: int = 8194
    ):
        self.securities = securities
        self.fields = fields
        self.start_date = start_date
        self.end_date = end_date
        self.host = host
        self.port = port
        logger.info(
            f"BloombergAPIDataSource initialized: {len(securities)} securities, "
            f"{len(fields)} fields, {start_date} to {end_date}"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Load data from Bloomberg Terminal API.

        Returns:
            pd.DataFrame: Data with dates and Bloomberg fields

        Raises:
            ImportError: If blpapi is not installed
            ConnectionError: If cannot connect to Bloomberg Terminal
            ValueError: If data cannot be retrieved
        """
        try:
            # Attempt to import Bloomberg API
            import blpapi
        except ImportError:
            error_msg = (
                "Bloomberg API (blpapi) not installed. "
                "Install with: pip install blpapi"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        # TODO: Implement Bloomberg API connection and data retrieval
        # This is a placeholder implementation
        raise NotImplementedError(
            "Bloomberg API integration not yet implemented. "
            "Please implement the connection and data retrieval logic, "
            "or use ExcelDataSource to load Bloomberg Excel exports."
        )

        # Example implementation structure (uncomment and adapt):
        """
        from blpapi import Session, SessionOptions

        # Create session
        sessionOptions = SessionOptions()
        sessionOptions.setServerHost(self.host)
        sessionOptions.setServerPort(self.port)
        session = Session(sessionOptions)

        if not session.start():
            raise ConnectionError("Failed to start Bloomberg session")

        if not session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open Bloomberg refdata service")

        # Request historical data
        service = session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")

        for security in self.securities:
            request.append("securities", security)

        for field in self.fields:
            request.append("fields", field)

        request.set("startDate", self.start_date.strftime("%Y%m%d"))
        request.set("endDate", self.end_date.strftime("%Y%m%d"))

        # Send request and process response
        session.sendRequest(request)

        # Parse response into DataFrame
        data_dict = {"Dates": []}
        for field in self.fields:
            data_dict[field] = []

        # ... parse Bloomberg response and populate data_dict ...

        df = pd.DataFrame(data_dict)
        df["Dates"] = pd.to_datetime(df["Dates"])

        self.validate_schema(df)
        session.stop()

        logger.info(f"Loaded {len(df)} rows from Bloomberg API")
        return df
        """


################################################################################
# CSV Data Source
################################################################################


class CSVDataSource(DataSource):
    """
    Load data from CSV files.

    Args:
        file_path: Path to CSV file
        date_column: Name of the date column (default: "Dates")
        date_format: Datetime format string (default: None = auto-detect)

    Example:
        >>> source = CSVDataSource("data/economic_data.csv")
        >>> df = source.load_data()
    """

    def __init__(
        self,
        file_path: str,
        date_column: str = "Dates",
        date_format: Optional[str] = None
    ):
        self.file_path = file_path
        self.date_column = date_column
        self.date_format = date_format
        logger.info(f"CSVDataSource initialized with file: {file_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
            pd.DataFrame: Data with dates and indicators

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If schema is invalid
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} rows from CSV file")

            # Ensure Dates column exists
            if self.date_column not in df.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in CSV")

            # Standardize column name
            if self.date_column != "Dates":
                df.rename(columns={self.date_column: "Dates"}, inplace=True)

            # Convert to datetime
            if self.date_format:
                df["Dates"] = pd.to_datetime(df["Dates"], format=self.date_format)
            else:
                df["Dates"] = pd.to_datetime(df["Dates"])

            self.validate_schema(df)

            return df

        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CSV file: {str(e)}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")


################################################################################
# Data Source Factory
################################################################################


class DataSourceFactory:
    """
    Factory class for creating data source instances.

    Simplifies data source creation by providing a unified interface.

    Example:
        >>> # Create from Excel
        >>> source = DataSourceFactory.create("excel", file_path="data.xlsx")
        >>>
        >>> # Create from CSV
        >>> source = DataSourceFactory.create("csv", file_path="data.csv")
        >>>
        >>> # Create from Bloomberg API (when implemented)
        >>> source = DataSourceFactory.create(
        ...     "bloomberg",
        ...     securities=["LF98TRUU Index"],
        ...     fields=["OAS"],
        ...     start_date=datetime(2020, 1, 1),
        ...     end_date=datetime(2020, 12, 31)
        ... )
    """

    @staticmethod
    def create(source_type: str, **kwargs) -> DataSource:
        """
        Create a data source instance.

        Args:
            source_type: Type of data source ("excel", "csv", "bloomberg")
            **kwargs: Arguments to pass to the data source constructor

        Returns:
            DataSource: Instance of the requested data source type

        Raises:
            ValueError: If source_type is not recognized
        """
        source_type = source_type.lower()

        if source_type == "excel":
            return ExcelDataSource(**kwargs)
        elif source_type == "csv":
            return CSVDataSource(**kwargs)
        elif source_type == "bloomberg":
            return BloombergAPIDataSource(**kwargs)
        elif source_type == "crypto":
            from _crypto_data_sources import CryptoExchangeDataSource
            return CryptoExchangeDataSource(**kwargs)
        elif source_type == "crypto_ws":
            from _crypto_data_sources import CryptoWebSocketDataSource
            return CryptoWebSocketDataSource(**kwargs)
        elif source_type == "crypto_agg":
            from _crypto_data_sources import CryptoAggregatorDataSource
            return CryptoAggregatorDataSource(**kwargs)
        else:
            raise ValueError(
                f"Unknown data source type: {source_type}. "
                f"Supported types: 'excel', 'csv', 'bloomberg', 'crypto', 'crypto_ws', 'crypto_agg'"
            )


################################################################################
# Configuration Helper
################################################################################


def load_data_from_config(config: Dict) -> pd.DataFrame:
    """
    Load data based on a configuration dictionary.

    This function provides a convenient way to load data from different
    sources based on configuration, making it easy to switch between
    data sources without changing code.

    Args:
        config: Dictionary with keys:
            - "source_type": "excel", "csv", or "bloomberg"
            - Additional keys depend on source type

    Returns:
        pd.DataFrame: Loaded data

    Example:
        >>> config = {
        ...     "source_type": "excel",
        ...     "file_path": "data/Economic_Data_2020_08_01.xlsx"
        ... }
        >>> df = load_data_from_config(config)
        >>>
        >>> # Or from environment/config file
        >>> import os
        >>> config = {
        ...     "source_type": os.getenv("DATA_SOURCE_TYPE", "excel"),
        ...     "file_path": os.getenv("DATA_FILE_PATH", "data/default.xlsx")
        ... }
        >>> df = load_data_from_config(config)
    """
    source_type = config.pop("source_type")
    source = DataSourceFactory.create(source_type, **config)
    return source.load_data()
