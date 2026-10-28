from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "examples" / "run_all_examples.sh"


def run_runner(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_dry_run_uses_full_data_and_100_epochs_by_default(tmp_path) -> None:
    output_dir = tmp_path / "not-created"

    result = run_runner("--dry-run", "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    stages = [
        "global",
        "supermodules",
        "report",
        "benchmark",
    ]
    positions = [result.stdout.index(f"DRY-RUN [{stage}]") for stage in stages]
    assert positions == sorted(positions)
    assert "--max-cells 0" in result.stdout
    assert "--epochs 100" in result.stdout
    assert "--classes all" in result.stdout
    assert "--gradient" not in result.stdout
    supermodule_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DRY-RUN [supermodules]")
    )
    report_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DRY-RUN [report]")
    )
    assert "--stop-on-error" in supermodule_line
    assert "supermodule_scfeaann.h5ad" in report_line
    assert not output_dir.exists()


def test_overrides_control_training_scope_and_optional_stages(tmp_path) -> None:
    output_dir = tmp_path / "overrides"

    result = run_runner(
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--max-cells",
        "13",
        "--epochs",
        "2",
        "--device",
        "cpu",
        "--classes",
        "1,8",
        "--with-gradient",
        "--skip-benchmark",
    )

    assert result.returncode == 0, result.stderr
    assert "--max-cells 13" in result.stdout
    assert "--epochs 2" in result.stdout
    assert "--device cpu" in result.stdout
    assert "CLASSES=1,8" in result.stdout
    assert "--classes" in result.stdout
    assert "--gradient" in result.stdout
    assert "DRY-RUN [benchmark]" not in result.stdout
    report_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DRY-RUN [report]")
    )
    assert "global_scfeaann.h5ad" in report_line
    assert "supermodule_scfeaann.h5ad" not in report_line
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--max-cells", "-1", "max-cells"),
        ("--epochs", "0", "epochs"),
        ("--classes", "", "classes"),
    ],
)
def test_invalid_arguments_fail_before_any_stage(
    option: str,
    value: str,
    message: str,
    tmp_path,
) -> None:
    result = run_runner(
        "--dry-run",
        "--output-dir",
        str(tmp_path / "invalid"),
        option,
        value,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_resume_skips_completed_stages_and_keeps_caches_local(
    tmp_path,
) -> None:
    output_dir = tmp_path / "resume"
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True)
    for artifact in (
        output_dir / "global_scfeaann.h5ad",
        output_dir / "supermodule_scfeaann.h5ad",
        report_dir / "report_manifest.json",
    ):
        artifact.write_text("complete\n", encoding="utf-8")

    result = run_runner(
        "--output-dir",
        str(output_dir),
        "--resume",
        "--skip-benchmark",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("SKIP [") == 3
    assert (output_dir / ".matplotlib").is_dir()
    assert (output_dir / ".cache").is_dir()
    assert (output_dir / ".numba").is_dir()
    summary = (output_dir / "run_summary.tsv").read_text(encoding="utf-8")
    assert summary.count("\tskipped\t") == 3
