"""
Tests for NEXUS v2 grounding rules.
Verifies the system enforces citation and tool evidence requirements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.llm.prompts import SYSTEM_PROMPT


class TestGroundingRules:
    """Verify grounding rules are present in system prompt."""

    def test_citation_rule(self):
        assert "CITE" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT

    def test_tool_requirement(self):
        assert "tool" in SYSTEM_PROMPT.lower()

    def test_refusal_rule(self):
        assert "don't have grounded" in SYSTEM_PROMPT.lower() or "grounded" in SYSTEM_PROMPT.lower()

    def test_all_tools_mentioned(self):
        assert "event_study_tool" in SYSTEM_PROMPT
        assert "volatility_tool" in SYSTEM_PROMPT
        assert "ecosystem_tool" in SYSTEM_PROMPT
        assert "price_tool" in SYSTEM_PROMPT

    def test_answer_format_required(self):
        assert "Evidence Used" in SYSTEM_PROMPT
        assert "Assumptions" in SYSTEM_PROMPT


class TestEvaluationPrompts:
    """Define expected tool usage for evaluation prompts."""

    EVAL_PROMPTS = [
        {
            "prompt": "Run CAR for NVDA around Oct 7 rules",
            "expected_tool": "event_study_tool",
        },
        {
            "prompt": "Compare NVDA vs TSM vs ASML correlation since 2023",
            "expected_tool": "ecosystem_tool",
        },
        {
            "prompt": "Is skew elevated in NVDA right now?",
            "expected_tool": "volatility_tool",
        },
        {
            "prompt": "What is NVDA stock price?",
            "expected_tool": "price_tool",
        },
    ]

    def test_eval_prompts_defined(self):
        assert len(self.EVAL_PROMPTS) >= 4

    def test_all_tools_covered(self):
        tools = set(p["expected_tool"] for p in self.EVAL_PROMPTS)
        assert "event_study_tool" in tools
        assert "volatility_tool" in tools
        assert "ecosystem_tool" in tools
        assert "price_tool" in tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
