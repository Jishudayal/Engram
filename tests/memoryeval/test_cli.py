"""Tests for memoryeval.cli — Click commands tested via CliRunner.

All tests use the ``memory`` backend so no real backend process is needed.
The ``run`` command is tested against real benchmark cases (InMemoryAdapter),
so scores are deterministic and well-understood.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from memoryeval.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memory_config(tmp_path: Path, **overrides: object) -> Path:
    """Write a minimal in-memory backend config to a temp file."""
    lines = ["backend:", "  name: memory"]
    for k, v in overrides.items():
        lines.append(f"  {k}: {v}")
    p = tmp_path / "config.yaml"
    p.write_text("\n".join(lines))
    return p


def _invoke(*args: str) -> object:
    """Invoke `memoryeval` with CliRunner."""
    runner = CliRunner()
    return runner.invoke(main, list(args), catch_exceptions=False)


# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    assert "benchmark" in result.output.lower() or "memoryeval" in result.output.lower()


def test_version_flag_exits_zero() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0
    assert "memoryeval" in result.output.lower() or "0." in result.output


def test_run_help_exits_zero() -> None:
    result = _invoke("run", "--help")
    assert result.exit_code == 0
    assert "--config" in result.output


def test_show_config_help_exits_zero() -> None:
    result = _invoke("show-config", "--help")
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `show-config` command
# ---------------------------------------------------------------------------


def test_show_config_valid_file_exits_zero(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    result = _invoke("show-config", str(cfg))
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_show_config_shows_backend_name(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    result = _invoke("show-config", str(cfg))
    assert "memory" in result.output


def test_show_config_invalid_yaml_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("backend: !!python/object/apply:os.system ['rm -rf /']")
    runner = CliRunner()
    result = runner.invoke(main, ["show-config", str(bad)], catch_exceptions=True)
    assert result.exit_code != 0


def test_show_config_missing_file_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["show-config", "/nonexistent/path.yaml"], catch_exceptions=True)
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `run` command — missing / invalid args
# ---------------------------------------------------------------------------


def test_run_without_config_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run"], catch_exceptions=True)
    assert result.exit_code != 0


def test_run_with_nonexistent_config_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--config", "/no/such/file.yaml"], catch_exceptions=True
    )
    assert result.exit_code != 0


def test_run_invalid_format_exits_nonzero(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--config", str(cfg), "--format", "xml"], catch_exceptions=True
    )
    assert result.exit_code != 0


def test_run_invalid_category_exits_nonzero(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg), "--category", "invalid_cat"],
        catch_exceptions=True,
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `run` command — plain format (fastest, no Rich dependencies)
# ---------------------------------------------------------------------------


def test_run_plain_exits_zero_on_all_pass(tmp_path: Path) -> None:
    """Temporal cases all pass on InMemoryAdapter → exit 0."""
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "plain",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


def test_run_plain_contains_backend_name(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "plain",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert "memory" in result.output


def test_run_plain_contains_category_in_output(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "plain",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert "temporal" in result.output


def test_run_plain_exits_one_when_failures_exist(tmp_path: Path) -> None:
    """Importance gap cases score 0.0 on InMemoryAdapter → exit 1."""
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "importance",
         "--format", "plain",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# `run` command — JSON format
# ---------------------------------------------------------------------------


def test_run_json_output_is_valid_json(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert isinstance(parsed, dict)


def test_run_json_has_schema_version(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert parsed["schema_version"] == "1"


def test_run_json_backend_name_matches_config(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert parsed["backend_name"] == "memory"


def test_run_json_category_filter_produces_one_category(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert len(parsed["categories"]) == 1
    assert parsed["categories"][0]["category"] == "temporal"


def test_run_json_two_category_filter(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--category", "multihop",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert len(parsed["categories"]) == 2
    names = {c["category"] for c in parsed["categories"]}
    assert names == {"temporal", "multihop"}


def test_run_json_overall_score_in_range(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert 0.0 <= parsed["overall_score"] <= 1.0


# ---------------------------------------------------------------------------
# `run` command — Rich format (smoke only; capture via no_color)
# ---------------------------------------------------------------------------


def test_run_rich_exits_zero_on_pass(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "rich",
         "--no-color",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_run_rich_contains_backend_name(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--category", "temporal",
         "--format", "rich",
         "--no-color",
         "--no-progress"],
        catch_exceptions=False,
    )
    assert "memory" in result.output


# ---------------------------------------------------------------------------
# `run` command — unknown backend raises UsageError
# ---------------------------------------------------------------------------


def test_run_unknown_backend_exits_nonzero(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("backend:\n  name: weaviate\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg), "--format", "plain", "--no-progress"],
        catch_exceptions=True,
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `run` command — full suite (no category filter)
# ---------------------------------------------------------------------------


def test_run_full_suite_produces_five_categories_in_json(tmp_path: Path) -> None:
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    parsed = json.loads(result.output)
    assert len(parsed["categories"]) == 5


def test_run_full_suite_exits_one_due_to_importance_gaps(tmp_path: Path) -> None:
    """The 5 importance gap cases score 0.0 → overall failures → exit 1."""
    cfg = _memory_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(cfg),
         "--format", "json",
         "--no-progress"],
        catch_exceptions=False,
    )
    # exit 1 because importance gap cases fail on InMemoryAdapter
    assert result.exit_code == 1
