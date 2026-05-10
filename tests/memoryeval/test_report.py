"""Tests for memoryeval.report.ReportRenderer.

All three output formats — Rich, plain, and JSON — are tested against
two hand-built BenchmarkReports:

  ``all_pass``  — every case passes (score=1.0, no errors)
  ``mixed``     — one failure (score=0.0) and one case with an error message
"""

from __future__ import annotations

import io
import json

import pytest
from memoryeval.report import ReportRenderer
from memoryeval.types import (
    BenchmarkCategory,
    BenchmarkReport,
    CaseResult,
    CategoryScore,
    HallucinationRisk,
)

# ---------------------------------------------------------------------------
# Shared test fixtures (built from types directly — no scorer needed)
# ---------------------------------------------------------------------------


def _make_result(
    name: str,
    category: BenchmarkCategory,
    score: float,
    *,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_name=name,
        category=category,
        score=score,
        passed=error is None and score >= 0.7,
        pass_threshold=0.7,
        error=error,
    )


def _make_report_all_pass() -> BenchmarkReport:
    """A report where all 10 temporal cases score 1.0."""
    results = [
        _make_result(f"temporal_case_{i}", BenchmarkCategory.TEMPORAL, 1.0)
        for i in range(10)
    ]
    cat = CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)
    return BenchmarkReport.from_category_scores("test-backend", [cat])


def _make_report_mixed() -> BenchmarkReport:
    """A report with one temporal pass, one failure, and one error.

    temporal:      8 pass (score=1.0) + 1 fail (score=0.0) + 1 error
    contradiction: 5 pass (score=0.85 each)
    """
    temporal_results = (
        [_make_result(f"t_pass_{i}", BenchmarkCategory.TEMPORAL, 1.0) for i in range(8)]
        + [_make_result("t_fail", BenchmarkCategory.TEMPORAL, 0.0)]
        + [_make_result("t_err",  BenchmarkCategory.TEMPORAL, 0.0, error="run: timed out after 30s")]
    )
    contra_results = [
        _make_result(f"c_{i}", BenchmarkCategory.CONTRADICTION, 0.85)
        for i in range(5)
    ]
    cats = [
        CategoryScore.from_results(BenchmarkCategory.TEMPORAL, temporal_results),
        CategoryScore.from_results(BenchmarkCategory.CONTRADICTION, contra_results),
    ]
    return BenchmarkReport.from_category_scores("qdrant-local", cats)


@pytest.fixture
def all_pass() -> BenchmarkReport:
    return _make_report_all_pass()


@pytest.fixture
def mixed() -> BenchmarkReport:
    return _make_report_mixed()


# ---------------------------------------------------------------------------
# to_json() — structure and correctness
# ---------------------------------------------------------------------------


def test_to_json_is_valid_json(all_pass) -> None:
    blob = ReportRenderer(all_pass).to_json()
    parsed = json.loads(blob)  # must not raise
    assert isinstance(parsed, dict)


def test_to_json_top_level_keys(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    expected = {"schema_version", "report_id", "run_at", "backend_name",
                "overall_score", "hallucination_risk", "categories"}
    assert set(parsed.keys()) == expected


def test_to_json_schema_version_is_one(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    assert parsed["schema_version"] == "1"


def test_to_json_backend_name(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    assert parsed["backend_name"] == "test-backend"


def test_to_json_overall_score(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    assert parsed["overall_score"] == 1.0


def test_to_json_hallucination_risk_is_string(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    assert isinstance(parsed["hallucination_risk"], str)
    assert parsed["hallucination_risk"] in {r.value for r in HallucinationRisk}


def test_to_json_categories_count(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    assert len(parsed["categories"]) == 1


def test_to_json_category_keys(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    cat = parsed["categories"][0]
    assert set(cat.keys()) == {"category", "score", "pass_rate", "results"}


def test_to_json_results_count(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    total = sum(len(c["results"]) for c in parsed["categories"])
    assert total == 10


def test_to_json_result_keys(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    result = parsed["categories"][0]["results"][0]
    assert set(result.keys()) == {
        "case_name", "score", "passed", "pass_threshold", "details", "error"
    }


def test_to_json_details_field_present_and_empty_dict_when_no_details(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    for cat in parsed["categories"]:
        for result in cat["results"]:
            assert "details" in result
            assert result["details"] == {}


def test_to_json_error_field_present_and_null_when_no_error(all_pass) -> None:
    parsed = json.loads(ReportRenderer(all_pass).to_json())
    for cat in parsed["categories"]:
        for result in cat["results"]:
            assert "error" in result
            assert result["error"] is None  # all pass, no errors


def test_to_json_error_field_non_null_when_error_exists(mixed) -> None:
    parsed = json.loads(ReportRenderer(mixed).to_json())
    temporal_cat = next(c for c in parsed["categories"] if c["category"] == "temporal")
    error_results = [r for r in temporal_cat["results"] if r["error"] is not None]
    assert len(error_results) == 1
    assert "timed out" in error_results[0]["error"]


def test_to_json_indent_parameter(all_pass) -> None:
    blob_2 = ReportRenderer(all_pass).to_json(indent=2)
    blob_4 = ReportRenderer(all_pass).to_json(indent=4)
    # Different indent → different formatting, same data
    assert json.loads(blob_2) == json.loads(blob_4)
    assert "    " in blob_4  # 4-space indent present
    assert "    " not in blob_2.split("\n")[1]  # 2-space indent, not 4


# ---------------------------------------------------------------------------
# to_plain() — structure and correctness
# ---------------------------------------------------------------------------


def _plain(report: BenchmarkReport) -> str:
    return ReportRenderer(report).to_plain()


def test_to_plain_returns_string(all_pass) -> None:
    assert isinstance(_plain(all_pass), str)


def test_to_plain_ends_with_newline(all_pass) -> None:
    assert _plain(all_pass).endswith("\n")


def test_to_plain_contains_backend_name(all_pass) -> None:
    assert "test-backend" in _plain(all_pass)


def test_to_plain_contains_run_at(all_pass) -> None:
    # Year 2026 should appear in the timestamp
    assert "2026" in _plain(all_pass)


def test_to_plain_contains_overall_score(all_pass) -> None:
    assert "1.000" in _plain(all_pass)


def test_to_plain_contains_risk_tier(all_pass) -> None:
    text = _plain(all_pass)
    # score=1.0 → LOW RISK
    assert "LOW RISK" in text


def test_to_plain_contains_category_names(all_pass) -> None:
    assert "temporal" in _plain(all_pass)


def test_to_plain_all_passed_message(all_pass) -> None:
    text = _plain(all_pass)
    assert "All" in text and "passed" in text


def test_to_plain_failure_list_when_failures(mixed) -> None:
    text = _plain(mixed)
    assert "failed" in text
    assert "t_fail" in text


def test_to_plain_error_message_in_failure_list(mixed) -> None:
    text = _plain(mixed)
    assert "timed out" in text


def test_to_plain_no_ansi_codes(all_pass) -> None:
    text = _plain(all_pass)
    assert "\x1b[" not in text  # no ANSI escape sequences


def test_to_plain_contains_pass_rate(all_pass) -> None:
    # 10/10 should appear for the all-pass report
    assert "10/10" in _plain(all_pass)


def test_to_plain_mixed_pass_rate(mixed) -> None:
    text = _plain(mixed)
    # 8 pass out of 10 temporal cases
    assert "8/10" in text


# ---------------------------------------------------------------------------
# to_rich() — smoke tests via captured console output
# ---------------------------------------------------------------------------


def _rich_output(report: BenchmarkReport, width: int = 120) -> str:
    buf = io.StringIO()
    con = __import__("rich").console.Console(file=buf, no_color=True, highlight=False, width=width)
    ReportRenderer(report).to_rich(console=con)
    return buf.getvalue()


def test_to_rich_does_not_raise(all_pass) -> None:
    _rich_output(all_pass)  # must not raise


def test_to_rich_contains_backend_name(all_pass) -> None:
    assert "test-backend" in _rich_output(all_pass)


def test_to_rich_contains_category_name(all_pass) -> None:
    assert "temporal" in _rich_output(all_pass)


def test_to_rich_shows_all_passed_when_no_failures(all_pass) -> None:
    out = _rich_output(all_pass)
    assert "passed" in out.lower()


def test_to_rich_shows_failure_list_when_failures(mixed) -> None:
    out = _rich_output(mixed)
    assert "failed" in out.lower()
    assert "t_fail" in out


def test_to_rich_shows_error_message(mixed) -> None:
    out = _rich_output(mixed)
    assert "timed out" in out


def test_to_rich_default_console_does_not_raise(all_pass, capsys) -> None:
    """to_rich() with no console argument must not raise (writes to real stdout)."""
    ReportRenderer(all_pass).to_rich()
    captured = capsys.readouterr()
    assert "test-backend" in captured.out


# ---------------------------------------------------------------------------
# Edge cases — empty report
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_report() -> BenchmarkReport:
    """A valid BenchmarkReport with no categories (scorer raised ValueError,
    but the renderer itself should handle this gracefully)."""
    return BenchmarkReport.from_category_scores("empty-backend", [])


def test_to_plain_empty_report_says_no_cases_ran(empty_report) -> None:
    text = ReportRenderer(empty_report).to_plain()
    assert "No cases ran" in text
    assert "passed" not in text


def test_to_rich_empty_report_says_no_cases_ran(empty_report) -> None:
    out = _rich_output(empty_report)
    assert "No cases ran" in out
    assert "All 0 cases passed" not in out


def test_to_json_empty_report_valid_json(empty_report) -> None:
    blob = json.loads(ReportRenderer(empty_report).to_json())
    assert blob["categories"] == []
    assert blob["schema_version"] == "1"


# ---------------------------------------------------------------------------
# Edge cases — general
# ---------------------------------------------------------------------------


def test_all_three_formats_parse_same_backend_name(all_pass) -> None:
    """All three formats must consistently reflect the backend name."""
    renderer = ReportRenderer(all_pass)
    json_data = json.loads(renderer.to_json())
    plain_text = renderer.to_plain()
    rich_buf = io.StringIO()
    con = __import__("rich").console.Console(file=rich_buf, no_color=True, width=120)
    renderer.to_rich(console=con)

    assert json_data["backend_name"] == "test-backend"
    assert "test-backend" in plain_text
    assert "test-backend" in rich_buf.getvalue()


def test_report_with_error_case_json_round_trip(mixed) -> None:
    """Error messages must survive JSON serialisation without corruption."""
    blob   = ReportRenderer(mixed).to_json()
    parsed = json.loads(blob)
    temporal = next(c for c in parsed["categories"] if c["category"] == "temporal")
    errors = [r["error"] for r in temporal["results"] if r["error"]]
    assert errors == ["run: timed out after 30s"]
