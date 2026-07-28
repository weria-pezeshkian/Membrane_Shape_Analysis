"""Tests for core/argument_parser.py::write_replay_file.

Regression coverage for a bug where adding --man broke replay-file
writing: write_replay_file iterates every registered CLI action and reads
its namespace value, but fire-and-exit actions (-h/--help, -v/--version,
--man) are SUPPRESS-defaulted, so argparse never sets a namespace attribute
for them unless invoked - getattr(ns, action.dest) raised AttributeError.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from CALM.core import argument_parser as arg_helper
from CALM.core.manual import add_manual


def test_write_replay_file_does_not_crash_on_suppress_defaulted_actions(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    parser.add_argument("-v", "--version", action="version", version="1.0")
    add_manual(parser, "test")

    ns = parser.parse_args(["-o", str(tmp_path)])
    arg_helper.write_replay_file(str(tmp_path / "replay.log"), parser, ns)

    assert (tmp_path / "replay.log").exists()


def test_write_replay_file_omits_replay_and_out_replay(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    ns = parser.parse_args(["-o", str(tmp_path), "--replay", "some_replay_file.log"])

    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    content = replay_path.read_text()
    assert "--replay" not in content
    assert "--out-replay" not in content


def test_write_replay_file_records_input_checksums(tmp_path: Path) -> None:
    traj = tmp_path / "traj.xtc"
    struct = tmp_path / "struct.gro"
    traj.write_bytes(b"trajectory bytes")
    struct.write_bytes(b"structure bytes")

    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj), "-s", str(struct)])

    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    content = replay_path.read_text()
    assert f"# Trajectory sha256: {arg_helper._sha256_of_file(str(traj))}" in content
    assert f"# Structure sha256: {arg_helper._sha256_of_file(str(struct))}" in content


def test_apply_replay_proceeds_silently_when_checksums_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"trajectory bytes")

    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj)])
    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    def _fail_if_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("input() should not be called when checksums match")

    monkeypatch.setattr("builtins.input", _fail_if_called)

    pre_ns = argparse.Namespace(replay=str(replay_path))
    result = arg_helper.apply_replay(parser, pre_ns, ["-o", str(tmp_path), "-f", str(traj)])
    assert result.trajectory == str(traj)


def test_apply_replay_aborts_when_checksum_mismatched_and_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"original bytes")

    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj)])
    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    traj.write_bytes(b"changed bytes - different trajectory content now")
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    pre_ns = argparse.Namespace(replay=str(replay_path))
    with pytest.raises(SystemExit):
        arg_helper.apply_replay(parser, pre_ns, ["-o", str(tmp_path), "-f", str(traj)])


def test_apply_replay_proceeds_when_checksum_mismatched_and_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"original bytes")

    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj)])
    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    traj.write_bytes(b"changed bytes - different trajectory content now")
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    pre_ns = argparse.Namespace(replay=str(replay_path))
    result = arg_helper.apply_replay(parser, pre_ns, ["-o", str(tmp_path), "-f", str(traj)])
    assert result.trajectory == str(traj)
