################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
"""
Configuration management for the BBG-Credit-Momentum application.

Loads configuration from environment variables, YAML files, or uses defaults.
Provides a centralized way to manage application settings.
"""
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("_config")

################################################################################
# Configuration Class
################################################################################


class Config:
    """
    Application configuration manager.

    Loads settings from (in order of precedence):
    1. Environment variables
    2. YAML configuration file
    3. Default values

    Usage:
        >>> config = Config()
        >>> print(config.get("model.type"))
        "XGBoost"
        >>>
        >>> # Access nested values
        >>> print(config.get("features.momentum_short_windows"))
        [5, 10, 15]
        >>>
        >>> # Provide default if key doesn't exist
        >>> print(config.get("missing.key", default="default_value"))
        "default_value"
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML config file (default: "config.yaml" in project root)
        """
        self.project_root = pathlib.Path(__file__).parent.absolute()

        if config_file is None:
            config_file = self.project_root / "config.yaml"

        self.config_file = pathlib.Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config or {}
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
                return self._default_config()
        else:
            logger.info(f"Config file not found: {self.config_file}. Using defaults.")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "data_source": {
                "type": "excel",
                "excel": {
                    "file_path": "data/Economic_Data_2020_08_01.xlsx",
                    "sheet_name": 0,
                    "date_column": "Dates",
                },
            },
            "model": {
                "type": "XGBoost",
                "estimators": 1000,
                "random_state": 42,
                "test_split": 0.20,
                "sequential": False,
            },
            "features": {
                "target": "LF98TRUU_Index_OAS",
                "momentum_columns": ["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
                "momentum_short_windows": [5, 10, 15],
                "momentum_long_window": 30,
                "forecast_horizons": [1, 3, 7, 15, 30],
            },
            "analysis": {
                "importance_threshold": 0.05,
                "usefulness_threshold": 0.2,
                "ppscore_threshold": 0.5,
                "max_forecast_days": 30,
                "cv_splits": 5,
            },
            "ui": {
                "default_start_date": "2012-08-08",
                "default_end_date": "2020-07-31",
                "chart_width": 1100,
                "image_width": 1000,
            },
            "logging": {
                "level": "INFO",
            },
            "performance": {
                "numexpr_threads": 16,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Checks environment variables first, then config file, then default.

        Args:
            key: Configuration key (use dot notation for nested keys, e.g., "model.type")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("model.type")
            "XGBoost"
            >>> config.get("model.missing_key", default="fallback")
            "fallback"
        """
        # Check environment variable first (uppercase, underscores)
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Navigate through config dict using dot notation
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value if value is not None else default

    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration."""
        source_type = self.get("data_source.type", default="excel")
        source_config = self.get(f"data_source.{source_type}", default={})
        return {"source_type": source_type, **source_config}

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.get("model.type", default="XGBoost"),
            "estimators": int(self.get("model.estimators", default=1000)),
            "random_state": self.get("model.random_state", default=42),
        }

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return {
            "target_col": self.get("features.target", default="LF98TRUU_Index_OAS"),
            "momentum_list": self.get(
                "features.momentum_columns",
                default=["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
            ),
            "momentum_X_days": self.get("features.momentum_short_windows", default=[5, 10, 15]),
            "momentum_Y_days": int(self.get("features.momentum_long_window", default=30)),
            "forecast_list": self.get("features.forecast_horizons", default=[1, 3, 7, 15, 30]),
            "split_percentage": float(self.get("model.test_split", default=0.20)),
            "sequential": bool(self.get("model.sequential", default=False)),
        }

    def get_bloomberg_config(self) -> Dict[str, Any]:
        """
        Get Bloomberg API configuration.

        Returns:
            Dict with Bloomberg API settings including:
            - securities: List of Bloomberg tickers
            - fields: List of fields to retrieve
            - start_date: Start date (as string)
            - end_date: End date (as string)
            - host: API host
            - port: API port
            - timeout: Connection timeout
            - max_retries: Maximum retry attempts
            - excel_fallback: Path to Excel fallback file

        Example:
            >>> config.get_bloomberg_config()
            {
                'securities': ['LF98TRUU Index', 'LUACTRUU Index'],
                'fields': ['OAS', 'PX_LAST'],
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'host': 'localhost',
                'port': 8194,
                'timeout': 30000,
                'max_retries': 3,
                'excel_fallback': 'data/bloomberg_export.xlsx'
            }
        """
        from datetime import datetime

        # Get securities list
        securities = self.get("data_source.bloomberg.securities", default=[
            "LF98TRUU Index",
            "LUACTRUU Index"
        ])

        # Get fields list
        fields = self.get("data_source.bloomberg.fields", default=["OAS", "PX_LAST"])

        # Get date range
        start_date = self.get("data_source.bloomberg.start_date", default="2020-01-01")
        end_date = self.get("data_source.bloomberg.end_date", default="2020-12-31")

        # Get connection settings
        host = self.get("data_source.bloomberg.host", default="localhost")
        port = int(self.get("data_source.bloomberg.port", default=8194))
        timeout = int(self.get("data_source.bloomberg.timeout", default=30000))
        max_retries = int(self.get("data_source.bloomberg.max_retries", default=3))

        # Get fallback settings
        excel_fallback = self.get(
            "data_source.bloomberg.excel_fallback",
            default="data/bloomberg_export.xlsx"
        )

        return {
            "securities": securities,
            "fields": fields,
            "start_date": start_date,
            "end_date": end_date,
            "host": host,
            "port": port,
            "timeout": timeout,
            "max_retries": max_retries,
            "excel_fallback": excel_fallback,
        }

    def set_numexpr_threads(self) -> None:
        """Set NUMEXPR_MAX_THREADS environment variable."""
        threads = str(self.get("performance.numexpr_threads", default=16))
        os.environ["NUMEXPR_MAX_THREADS"] = threads
        logger.info(f"Set NUMEXPR_MAX_THREADS to {threads}")


################################################################################
# Global Config Instance
################################################################################

# Create a global config instance for easy import
config = Config()

# Set performance settings
config.set_numexpr_threads()


################################################################################
# Utility Functions
################################################################################


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config: Global configuration object

    Example:
        >>> from _config import get_config
        >>> cfg = get_config()
        >>> print(cfg.get("model.type"))
    """
    return config


def reload_config(config_file: Optional[str] = None) -> Config:
    """
    Reload configuration from file.

    Useful if config file has been updated during runtime.

    Args:
        config_file: Path to config file (default: "config.yaml")

    Returns:
        Config: New configuration instance
    """
    global config
    config = Config(config_file)
    config.set_numexpr_threads()
    return config
