import subprocess
from pathlib import Path

import scripts.run_experiments as run_experiments


def _write_completed_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "best.pt").write_bytes(b"checkpoint")
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (output_dir / "config.resolved.yaml").write_text("name: done\n", encoding="utf-8")


def test_main_continues_when_failed_training_left_complete_artifacts(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "first.yaml").write_text("name: first\n", encoding="utf-8")
    (config_dir / "second.yaml").write_text("name: second\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    calls = []

    def fake_run(command, *, check):
        calls.append(command)
        config_name = Path(command[3]).stem
        _write_completed_artifacts(tmp_path / "outputs" / config_name)
        if config_name == "first":
            raise subprocess.CalledProcessError(0xC0000409, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_experiments.subprocess, "run", fake_run)

    assert run_experiments.main(["--config-dir", str(config_dir), "--device", "cuda"]) == 0

    assert [Path(call[3]).stem for call in calls] == ["first", "second"]
    assert "[warn] training command failed after complete artifacts were written" in capsys.readouterr().out
