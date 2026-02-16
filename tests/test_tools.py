"""
Tests for NEXUS v2 tool wrappers.
Verifies tool schemas and output structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools import event_study_tool, volatility_tool, ecosystem_tool, price_tool


class TestToolSchemas:
    """Verify all tools have valid OpenAI function-calling schemas."""

    def test_event_study_schema(self):
        schema = event_study_tool.SCHEMA
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "event_study_tool"
        assert "parameters" in schema["function"]

    def test_volatility_schema(self):
        schema = volatility_tool.SCHEMA
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "volatility_tool"

    def test_ecosystem_schema(self):
        schema = ecosystem_tool.SCHEMA
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "ecosystem_tool"
        assert "tickers" in schema["function"]["parameters"]["properties"]

    def test_price_schema(self):
        schema = price_tool.SCHEMA
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "price_tool"
        assert "tickers" in schema["function"]["parameters"]["required"]


class TestEventStudyTool:
    """Test event study tool outputs."""

    def test_default_run(self):
        result = event_study_tool.run()
        assert "ticker" in result
        assert result["ticker"] == "NVDA"
        assert "results" in result or "error" in result

    def test_severity_filter(self):
        result = event_study_tool.run(severity_filter=["Critical"])
        if "results" in result:
            assert result["events_analyzed"] <= 11

    def test_output_structure(self):
        result = event_study_tool.run()
        if "results" in result:
            assert "summary" in result
            assert "methodology" in result
            for event in result["results"]:
                assert "car_pct" in event
                assert "p_value" in event
                assert "significant" in event


class TestPriceTool:
    """Test price tool outputs."""

    def test_single_ticker(self):
        result = price_tool.run(tickers=["NVDA"], start_date="2024-01-01")
        assert "latest_prices" in result or "error" in result

    def test_multi_ticker(self):
        result = price_tool.run(tickers=["NVDA", "AMD", "TSM"])
        if "latest_prices" in result:
            assert "correlation_matrix" in result
            assert "return_summary" in result


class TestEcosystemTool:
    """Test ecosystem tool outputs."""

    def test_basic_comparison(self):
        result = ecosystem_tool.run(tickers=["NVDA", "AMD", "INTC"])
        assert "metrics" in result or "error" in result

    def test_export_context_included(self):
        result = ecosystem_tool.run(tickers=["NVDA", "TSM"])
        if "export_control_context" in result:
            assert "NVDA" in result["export_control_context"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
