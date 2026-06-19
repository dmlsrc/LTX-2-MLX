"""Tests for the console-log sidecar tee (LTX_2_MLX/console_log.py).

The interesting case is the cursor-aware cleaning: stacked progress bars redraw
their rows with ANSI cursor moves, and the tee must land each on its own clean
line (no escapes, no collapse) -- so it is exercised against the real PhaseBar.
"""

import io
import sys
import threading

from LTX_2_MLX.console_log import _ScreenTee, _Tee, capture_console, format_command
from LTX_2_MLX.progress import StackedPhaseBars


def test_format_command_keeps_flag_and_value_on_one_line():
    cmd = format_command(
        ["/usr/bin/ltx2mlx", "a prompt", "--pipeline", "distilled",
         "--height", "448", "--fast-mode", "--width", "768"]
    )
    lines = cmd.split(" \\\n  ")
    assert lines[0] == "ltx2mlx"
    assert lines[1] == "'a prompt'"          # positional on its own line, quoted
    assert "--pipeline distilled" in lines   # flag + value together
    assert "--height 448" in lines
    assert "--width 768" in lines
    assert "--fast-mode" in lines            # bare flag on its own line


def test_single_bar_carriage_return_collapses_to_final_line():
    buf = io.StringIO()
    screen = _ScreenTee(buf)
    screen.feed("loading: 1/3\rloading: 2/3\rloading: 3/3\n")
    screen.close()
    assert buf.getvalue().strip() == "loading: 3/3"


def test_stacked_bars_land_on_separate_clean_lines():
    buf = io.StringIO()
    screen = _ScreenTee(buf)
    old = sys.stderr
    sys.stderr = _Tee(io.StringIO(), screen, threading.Lock())
    try:
        print("=== Running ===", file=sys.stderr)
        with StackedPhaseBars() as bars:
            a = bars.add(total=2, desc="VAE chunks", unit="chunk")
            b = bars.add(total=4, desc="VT encode", unit="frame")
            for _ in range(2):
                a.update(1)
                a.close()
                b.update(2)
            b.close()
        print("done", file=sys.stderr)
    finally:
        sys.stderr = old
        screen.close()

    out = buf.getvalue()
    assert out.count("VAE chunks") == 1, out
    assert out.count("VT encode") == 1, out
    assert "done" in out
    assert "\x1b" not in out  # no ANSI escapes leaked into the file


def test_capture_console_writes_command_header_and_both_streams(tmp_path):
    path = tmp_path / "run_console.txt"
    real_out, real_err = sys.stdout, sys.stderr
    with capture_console(str(path), "ltx2mlx --foo bar"):
        print("hello from stdout")
        print("warn line", file=sys.stderr)
    # streams restored
    assert sys.stdout is real_out and sys.stderr is real_err
    text = path.read_text()
    assert text.startswith("ltx2mlx --foo bar")
    assert "hello from stdout" in text
    assert "warn line" in text
