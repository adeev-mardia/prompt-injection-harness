import io
import sys
import tempfile
from pathlib import Path

from prompt_injection_harness.cli import main


def _run(argv, capsys):
    exit_code = main(argv)
    captured = capsys.readouterr()
    return exit_code, captured.out


def test_cli_benchmark_naive_only(capsys):
    code, out = _run(["benchmark", "--pipeline", "naive"], capsys)
    assert code == 0
    assert "naive_concat" in out
    assert "Overall ASR" in out


def test_cli_benchmark_all(capsys):
    code, out = _run(["benchmark", "--pipeline", "all"], capsys)
    assert code == 0
    assert "naive_concat" in out
    assert "sanitizing" in out


def test_cli_detect_flags_poisoned_file(capsys, tmp_path):
    from prompt_injection_harness.payloads import PAYLOAD_LIBRARY

    f = tmp_path / "poisoned.txt"
    f.write_text(PAYLOAD_LIBRARY[0].render("ZZZ99999"))

    code, out = _run(["detect", str(f)], capsys)
    assert code == 0
    assert "likely_injection: True" in out


def test_cli_detect_does_not_flag_clean_file(capsys, tmp_path):
    f = tmp_path / "clean.txt"
    f.write_text("This is a perfectly ordinary sentence about gardening.")

    code, out = _run(["detect", str(f)], capsys)
    assert code == 0
    assert "likely_injection: False" in out


def test_cli_detect_missing_file(capsys):
    code, out = _run(["detect", "/no/such/file.txt"], capsys)
    assert code == 1
