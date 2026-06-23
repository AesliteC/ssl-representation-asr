from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_experiments_script_streams_utf8_logs():
    script = (ROOT / "scripts" / "run_full_experiments.ps1").read_text(encoding="utf-8")

    assert "Tee-Object" not in script
    assert "Add-Content" not in script
    assert "--no-capture-output" in script
    assert "UTF8Encoding($false)" in script
    assert "Write-ExperimentLogLine" in script
