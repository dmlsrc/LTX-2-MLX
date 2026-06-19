"""Tee the console (stdout + stderr) to a sidecar log file, with the command.

`capture_console(path, command)` installs a tee on `sys.stdout`/`sys.stderr`
for the duration of a `with` block: every byte still reaches the real terminal
unchanged (so the live `progress.PhaseBar` redraws keep working), and a cleaned
copy is written to `path`.

The cleaning is the whole trick. Progress bars redraw their rows with ANSI
cursor moves (`\\033[nA` up, `\\033[nB` down, `\\033[J` erase-to-end) plus a
carriage return; a naive capture either keeps the escapes (garbage in a text
file) or strips them (stacked bars then overwrite the same buffered line and
collapse to one). So the file side runs a tiny cursor-aware screen model -- a
sliding window of row strings with a (row, col) cursor that honors `\\r`, `\\n`,
and those few cursor escapes -- so each bar lands on its own row and finalizes
as one clean line. Styling escapes (colors, etc.) are dropped.

No existing `print()` is touched: the redirection happens at the stream level.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading

# CSI sequences we care about positionally (A/B/J); everything else (SGR colors,
# K, etc.) is matched so it can be dropped without leaking escapes into the file.
_CSI = re.compile(r"\x1b\[([0-9]*)([A-Za-z])")

# Rows kept "live" (still reachable by a cursor-up) before flushing to disk.
# Comfortably larger than any progress-bar stack, so cursor-ups never reach a
# row that has already been written out.
_WINDOW = 64


class _ScreenTee:
    """Cleaned, cursor-aware sink writing to an open file handle."""

    def __init__(self, fh):
        self._fh = fh
        self._rows: list[str] = [""]
        self._row = 0
        self._col = 0
        self._pending = ""  # partial escape carried across writes

    def feed(self, text: str) -> None:
        data = self._pending + text
        self._pending = ""
        i, n = 0, len(data)
        while i < n:
            ch = data[i]
            if ch == "\x1b":
                m = _CSI.match(data, i)
                if not m:
                    # Incomplete escape at the end of this chunk; wait for more.
                    self._pending = data[i:]
                    return
                arg, cmd = m.group(1), m.group(2)
                count = int(arg) if arg else 1
                if cmd == "A":  # cursor up
                    self._row = max(0, self._row - count)
                elif cmd == "B":  # cursor down
                    self._row += count
                    self._ensure(self._row)
                elif cmd == "J":  # erase from cursor to end of screen
                    self._rows[self._row] = self._rows[self._row][: self._col]
                    del self._rows[self._row + 1 :]
                # other CSI (colors, line-erase, ...) -> drop, no positional effect
                i = m.end()
                continue
            if ch == "\r":
                self._col = 0
            elif ch == "\n":
                self._row += 1
                self._col = 0
                self._ensure(self._row)
                self._drain()
            else:
                row = self._rows[self._row]
                if self._col < len(row):
                    row = row[: self._col] + ch + row[self._col + 1 :]
                else:
                    row = row + " " * (self._col - len(row)) + ch
                self._rows[self._row] = row
                self._col += 1
            i += 1

    def _ensure(self, idx: int) -> None:
        while idx >= len(self._rows):
            self._rows.append("")

    def _drain(self) -> None:
        # Flush rows too far above the cursor to ever be redrawn.
        while self._row > _WINDOW:
            self._fh.write(self._rows.pop(0).rstrip() + "\n")
            self._row -= 1

    def close(self) -> None:
        # Drop trailing blank rows left parked below a finished bar stack.
        while len(self._rows) > 1 and self._rows[-1] == "":
            self._rows.pop()
        for row in self._rows:
            self._fh.write(row.rstrip() + "\n")
        self._rows = [""]
        self._row = self._col = 0
        self._fh.flush()


class _Tee:
    """Mirror writes to the real stream (raw) and the cleaned file sink."""

    def __init__(self, real, screen: _ScreenTee, lock: threading.Lock):
        self._real = real
        self._screen = screen
        self._lock = lock

    def write(self, data: str) -> int:
        with self._lock:
            self._real.write(data)
            self._screen.feed(data)
        return len(data)

    def flush(self) -> None:
        self._real.flush()

    def isatty(self) -> bool:
        return self._real.isatty()

    def __getattr__(self, name):
        # Delegate fileno(), encoding, etc. to the real stream.
        return getattr(self._real, name)


def start_console_capture(path: str, command: str):
    """Install the stdout/stderr tee now; return a no-arg teardown callable.

    `command` is written as a header before any captured output, which is then
    flushed incrementally. Returning a teardown (rather than only a context
    manager) lets the caller do `atexit.register(start_console_capture(...))`
    and capture the rest of the process without wrapping the call site.
    """
    fh = open(path, "w", encoding="utf-8")
    fh.write(command.rstrip() + "\n\n")
    fh.flush()
    screen = _ScreenTee(fh)
    lock = threading.Lock()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, screen, lock)
    sys.stderr = _Tee(old_err, screen, lock)

    def teardown() -> None:
        sys.stdout, sys.stderr = old_out, old_err
        with lock:
            screen.close()
        fh.close()

    return teardown


@contextlib.contextmanager
def capture_console(path: str, command: str):
    """Context-manager form of `start_console_capture` (used in tests)."""
    teardown = start_console_capture(path, command)
    try:
        yield
    finally:
        teardown()
