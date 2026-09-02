from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading

from .introspect import FieldSpec


def build_argv(argv_prefix: list[str], specs: list[FieldSpec], values: dict[str, object]) -> list[str]:
    """CALM's own argv, from one command's field specs and the GUI's current widget values.

    `values` is keyed by dest: a bool for "bool"/"bool_optional" fields, a
    list[str] for "multichoice"/"multi" fields, a str for everything else
    (including numeric fields - the real CLI's own `type=` callable does
    the actual conversion/validation when the subprocess runs, not the
    GUI). An empty/unset optional field (empty string, empty list, or
    missing from `values` entirely) is omitted from argv - never passed as
    `--flag ""`, so a `default=None` flag stays genuinely unset, the same
    way `write_replay_file` already treats an unset value.
    """
    argv = ["CALM", *argv_prefix]
    for spec in specs:
        value = values.get(spec.dest)
        if spec.kind == "bool":
            if value:
                argv.append(spec.flag)
        elif spec.kind == "bool_optional":
            long_name = spec.flag.lstrip("-")
            argv.append(f"--{long_name}" if value else f"--no-{long_name}")
        elif spec.kind in ("multichoice", "multi"):
            if value:
                assert isinstance(value, (list, tuple))
                argv.append(spec.flag)
                argv.extend(str(v) for v in value)
        else:
            if value is not None and str(value) != "":
                argv.append(spec.flag)
                argv.append(str(value))
    return argv


class CommandRunner:
    """Runs one CALM CLI invocation as a real subprocess (not an in-process function call - several
    CLI entry points call sys.exit(), which would kill the GUI itself if called directly), streaming
    its merged stdout/stderr into a thread-safe queue the GUI polls from its own main loop (Tkinter
    widgets may only be touched from the main thread - see gui/app.py's poll loop).
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, argv: list[str], cwd: str) -> None:
        """Launch `argv` (as built by `build_argv`) with `cwd` as its working directory."""
        if self.is_running():
            raise RuntimeError("A command is already running.")

        # A new process group (POSIX) / process group (Windows), so `stop`
        # can kill the whole group, not just this immediate child - a
        # --Workers N run's own worker processes would otherwise be
        # orphaned by killing only the parent. subprocess.CREATE_NEW_PROCESS_GROUP
        # only exists on Windows, hence the type: ignore on that branch.
        if os.name == "posix":
            self.process = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
        else:
            self.process = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
            )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line.rstrip("\n"))
        self.process.wait()
        self.output_queue.put(None)  # sentinel: the process has finished

    def stop(self) -> None:
        """Kill the running process's whole process group, waiting briefly before escalating to
        a hard kill if it doesn't exit on its own."""
        if not self.is_running():
            return
        assert self.process is not None
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except (ProcessLookupError, OSError):
            pass

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except (ProcessLookupError, OSError):
                pass
