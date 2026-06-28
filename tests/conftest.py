"""Shared pytest fixtures.

The full coverage report is computed once per session because reading every
processed JSON + scanning the Chroma collection is a few hundred milliseconds
and we don't want each assertion to re-do that work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.kb_coverage import CoverageReport, compute_coverage


@pytest.fixture(scope="session")
def coverage_report() -> CoverageReport:
    """Structural coverage report (no API calls)."""
    return compute_coverage(run_retrieval=False)
