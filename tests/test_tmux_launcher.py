import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.start_tmux_experiments import (
    build_pipeline_command,
    build_powershell_pipeline_command,
    build_tmux_command,
)


class TmuxLauncherTests(unittest.TestCase):
    def test_build_pipeline_command_contains_full_experiment_steps(self):
        command = build_pipeline_command(
            python_executable="python",
            device="cuda",
            log_path=Path("logs/full_experiments.log"),
        )

        self.assertIn("scripts/prepare_experiment_assets.py --device cuda", command)
        self.assertIn("scripts/run_experiments.py --config-dir configs/experiments --device cuda --skip-existing", command)
        self.assertIn("scripts/evaluate_experiments.py --config-dir configs/experiments --device cuda --skip-missing", command)
        self.assertIn("scripts/summarize.py --outputs-dir outputs --output results/summary.csv", command)
        self.assertIn("logs/full_experiments.log", command)

    def test_build_powershell_pipeline_command_uses_ps1_runner(self):
        command = build_powershell_pipeline_command(
            script_path=Path("scripts/run_full_experiments.ps1"),
            device="cuda",
        )

        self.assertIn("powershell", command)
        self.assertIn("-File scripts/run_full_experiments.ps1", command)
        self.assertIn("-Device cuda", command)

    def test_build_tmux_command_uses_session_name_and_tmux_path(self):
        command = build_tmux_command(
            tmux_executable="C:/tmux/tmux.exe",
            session="ssl_asr_full",
            pipeline_command="echo hi",
        )

        self.assertEqual(command[:4], ["C:/tmux/tmux.exe", "new-session", "-d", "-s"])
        self.assertIn("ssl_asr_full", command)
        self.assertEqual(command[-1], "echo hi")


if __name__ == "__main__":
    unittest.main()
