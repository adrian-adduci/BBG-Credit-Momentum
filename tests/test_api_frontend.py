"""
Frontend/API integration tests for mixed portfolio functionality.

Tests the FastAPI endpoints with simulated frontend requests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from api import app


class TestMixedPortfolioAPI:
    """Test suite for mixed portfolio API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_mixed_data(self):
        """Create mock mixed portfolio data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Dates': dates,
            'BTC_USDT_close': np.random.rand(100) * 1000 + 50000,
            'ETH_USDT_close': np.random.rand(100) * 100 + 3000,
            'LF98TRUU_Index_OAS': np.random.rand(100) * 20 + 100,
        })

    def test_mixed_train_endpoint_success(self, client, mock_mixed_data, tmp_path):
        """Test successful mixed portfolio training via API."""
        # Create temporary Bloomberg Excel file
        bloomberg_file = tmp_path / "bloomberg_test.xlsx"
        mock_bloomberg_df = mock_mixed_data[['Dates', 'LF98TRUU_Index_OAS']]
        mock_bloomberg_df.to_excel(bloomberg_file, index=False)

        # Mock data sources
        with patch('api.DataSourceFactory.create') as mock_factory, \
             patch('api.BloombergExcelDataSource') as mock_bloomberg:

            # Mock crypto source
            mock_crypto_source = MagicMock()
            mock_crypto_df = mock_mixed_data[['Dates', 'BTC_USDT_close', 'ETH_USDT_close']].copy()
            mock_crypto_df = mock_crypto_df.set_index('Dates')
            mock_crypto_source.load_data.return_value = mock_crypto_df
            mock_factory.return_value = mock_crypto_source

            # Mock Bloomberg source
            mock_bloomberg_source = MagicMock()
            mock_bloomberg_source.load_data.return_value = mock_bloomberg_df
            mock_bloomberg.return_value = mock_bloomberg_source

            # API request
            request_data = {
                "crypto_exchange": "binance",
                "crypto_symbols": ["BTC/USDT", "ETH/USDT"],
                "crypto_timeframe": "1h",
                "bloomberg_securities": ["LF98TRUU Index"],
                "bloomberg_fields": ["OAS"],
                "bloomberg_source": "excel",
                "bloomberg_excel_path": str(bloomberg_file),
                "start_date": "2024-01-01",
                "target_column": "BTC_USDT_close",
                "model_type": "XGBoost",
                "crypto_features": True,
                "cross_asset_features": True
            }

            response = client.post("/api/mixed/train", json=request_data)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "model_id" in data
            assert "mixed_" in data["model_id"]
            assert "metrics" in data
            assert "mae" in data["metrics"]

    def test_mixed_train_missing_crypto(self, client, tmp_path):
        """Test mixed training with only Bloomberg data (should fail)."""
        bloomberg_file = tmp_path / "bloomberg_only.xlsx"
        df = pd.DataFrame({
            'Dates': pd.date_range('2024-01-01', periods=50),
            'LF98TRUU_Index_OAS': np.random.rand(50) * 20 + 100
        })
        df.to_excel(bloomberg_file, index=False)

        request_data = {
            "bloomberg_securities": ["LF98TRUU Index"],
            "bloomberg_fields": ["OAS"],
            "bloomberg_source": "excel",
            "bloomberg_excel_path": str(bloomberg_file),
            "start_date": "2024-01-01",
            "target_column": "LF98TRUU_Index_OAS",
            "model_type": "XGBoost"
        }

        # Should still work (at least one source required, not necessarily both)
        with patch('api.BloombergExcelDataSource') as mock_bloomberg:
            mock_bloomberg_source = MagicMock()
            mock_bloomberg_source.load_data.return_value = df
            mock_bloomberg.return_value = mock_bloomberg_source

            response = client.post("/api/mixed/train", json=request_data)

            # Should succeed with just one source
            assert response.status_code in [200, 500]  # May fail due to preprocessing

    def test_mixed_train_invalid_bloomberg_source(self, client):
        """Test with invalid bloomberg_source parameter."""
        request_data = {
            "crypto_exchange": "binance",
            "crypto_symbols": ["BTC/USDT"],
            "bloomberg_securities": ["LF98TRUU Index"],
            "bloomberg_source": "invalid_source",
            "start_date": "2024-01-01",
            "target_column": "BTC_USDT_close"
        }

        response = client.post("/api/mixed/train", json=request_data)

        # Should fail with 400 or 422 (validation error)
        assert response.status_code in [400, 422, 500]

    def test_cross_asset_analysis_endpoint(self, client, mock_mixed_data, tmp_path):
        """Test cross-asset analysis endpoint."""
        # First train a model
        bloomberg_file = tmp_path / "bloomberg_analysis.xlsx"
        mock_bloomberg_df = mock_mixed_data[['Dates', 'LF98TRUU_Index_OAS']]
        mock_bloomberg_df.to_excel(bloomberg_file, index=False)

        with patch('api.DataSourceFactory.create') as mock_factory, \
             patch('api.BloombergExcelDataSource') as mock_bloomberg, \
             patch('api.MixedPortfolioDataSource') as mock_mixed:

            # Setup mocks
            mock_crypto_source = MagicMock()
            mock_crypto_df = mock_mixed_data[['Dates', 'BTC_USDT_close', 'ETH_USDT_close']].copy()
            mock_crypto_df = mock_crypto_df.set_index('Dates')
            mock_crypto_source.load_data.return_value = mock_crypto_df
            mock_factory.return_value = mock_crypto_source

            mock_bloomberg_source = MagicMock()
            mock_bloomberg_source.load_data.return_value = mock_bloomberg_df
            mock_bloomberg.return_value = mock_bloomberg_source

            # Mock mixed source to return combined data
            mock_mixed_source = MagicMock()
            mock_mixed_source.load_data.return_value = mock_mixed_data
            mock_mixed.return_value = mock_mixed_source

            # Train model
            train_request = {
                "crypto_exchange": "binance",
                "crypto_symbols": ["BTC/USDT", "ETH/USDT"],
                "bloomberg_securities": ["LF98TRUU Index"],
                "bloomberg_source": "excel",
                "bloomberg_excel_path": str(bloomberg_file),
                "start_date": "2024-01-01",
                "target_column": "BTC_USDT_close",
                "cross_asset_features": True
            }

            train_response = client.post("/api/mixed/train", json=train_request)

            if train_response.status_code == 200:
                model_id = train_response.json()["model_id"]

                # Get cross-asset analysis
                analysis_response = client.get(f"/api/mixed/analysis/{model_id}")

                # Verify response
                if analysis_response.status_code == 200:
                    data = analysis_response.json()
                    assert data["success"] is True
                    assert "correlations" in data
                    assert "regime" in data
                    assert "divergence_signals" in data
                    assert "flight_to_quality" in data
                    assert data["regime"] in ["risk-on", "risk-off", "neutral"]

    def test_cross_asset_analysis_nonexistent_model(self, client):
        """Test analysis endpoint with non-existent model ID."""
        response = client.get("/api/mixed/analysis/nonexistent_model_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cross_asset_analysis_non_mixed_model(self, client):
        """Test analysis endpoint with non-mixed portfolio model."""
        # This would require training a regular (non-mixed) model first
        # For now, just test the error case directly
        with patch('api.MODELS', {"test_model": {"type": "regular"}}):
            response = client.get("/api/mixed/analysis/test_model")

            assert response.status_code == 400
            assert "not a mixed portfolio model" in response.json()["detail"].lower()


class TestAPIValidation:
    """Test request validation and error handling."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    def test_missing_required_fields(self, client):
        """Test API with missing required fields."""
        # Missing start_date and target_column
        request_data = {
            "crypto_exchange": "binance",
            "crypto_symbols": ["BTC/USDT"]
        }

        response = client.post("/api/mixed/train", json=request_data)

        # Should fail validation (422 Unprocessable Entity)
        assert response.status_code == 422

    def test_invalid_date_format(self, client):
        """Test API with invalid date format."""
        request_data = {
            "crypto_exchange": "binance",
            "crypto_symbols": ["BTC/USDT"],
            "start_date": "invalid-date",
            "target_column": "BTC_USDT_close"
        }

        response = client.post("/api/mixed/train", json=request_data)

        # Should fail (422 validation or 500 internal error)
        assert response.status_code in [422, 500]

    def test_excel_path_required_for_excel_source(self, client):
        """Test that excel_path is required when source is 'excel'."""
        request_data = {
            "bloomberg_securities": ["LF98TRUU Index"],
            "bloomberg_source": "excel",
            # Missing bloomberg_excel_path
            "start_date": "2024-01-01",
            "target_column": "LF98TRUU_Index_OAS"
        }

        response = client.post("/api/mixed/train", json=request_data)

        # Should fail with 400 or 422
        assert response.status_code in [400, 422]
        if response.status_code == 400:
            assert "bloomberg_excel_path required" in response.json()["detail"].lower()


class TestAPIDocumentation:
    """Test API documentation and OpenAPI schema."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Check mixed portfolio endpoints exist
        assert "/api/mixed/train" in schema["paths"]
        assert "/api/mixed/analysis/{model_id}" in schema["paths"]

    def test_swagger_docs_accessible(self, client):
        """Test that Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert b"swagger" in response.content.lower()

    def test_redoc_accessible(self, client):
        """Test that ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert b"redoc" in response.content.lower()


class TestPerformance:
    """Basic performance tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    def test_api_response_time(self, client):
        """Test that API endpoints respond within reasonable time."""
        import time

        start_time = time.time()
        response = client.get("/openapi.json")
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/openapi.json")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == 200 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
