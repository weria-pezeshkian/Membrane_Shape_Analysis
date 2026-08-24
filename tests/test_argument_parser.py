"""Tests for core/argument_parser.py.

write_replay_file coverage is regression coverage for a bug where adding
--man broke replay-file writing: write_replay_file iterates every
registered CLI action and reads its namespace value, but fire-and-exit
actions (-h/--help, -v/--version, --man) are SUPPRESS-defaulted, so argparse
never sets a namespace attribute for them unless invoked - getattr(ns,
action.dest) raised AttributeError. lipids_species_token covers --lipids'
'RESNAME' / 'RESNAME:NAME1,NAME2,...' token format validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from CALM.core import argument_parser as arg_helper
from CALM.core.manual import add_manual


def _fail_if_called(*args: object, **kwargs: object) -> str:
    raise AssertionError("input() should not be called when checksums match")


def test_write_replay_file_does_not_crash_on_suppress_defaulted_actions(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)
    parser.add_argument("-v", "--version", action="version", version="1.0")
    add_manual(parser, "test")

    ns = parser.parse_args(["-o", str(tmp_path)])
    arg_helper.write_replay_file(str(tmp_path / "replay.log"), parser, ns)

    assert (tmp_path / "replay.log").exists()


def test_write_replay_file_omits_replay_and_out_replay(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)
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
    arg_helper.add_build_arguments(parser, require_inputs=False)
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
    arg_helper.add_build_arguments(parser, require_inputs=False)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj)])
    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    monkeypatch.setattr("builtins.input", _fail_if_called)

    pre_ns = argparse.Namespace(replay=str(replay_path))
    result = arg_helper.apply_replay(parser, pre_ns, ["-o", str(tmp_path), "-f", str(traj)])
    assert result.trajectory == str(traj)


def test_remove_tmd_parses_bare_as_true_and_with_value_as_the_string(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)

    assert parser.parse_args(["-o", str(tmp_path)]).remove_tmd is False
    assert parser.parse_args(["-o", str(tmp_path), "--Remove-TMD"]).remove_tmd is True
    ns = parser.parse_args(["-o", str(tmp_path), "--Remove-TMD", "name BB SC1"])
    assert ns.remove_tmd == "name BB SC1"


def test_validate_rotation_args_requires_center_only_for_bare_remove_tmd(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)

    bare = parser.parse_args(["-o", str(tmp_path), "--Remove-TMD"])
    with pytest.raises(SystemExit):
        arg_helper.validate_rotation_args(parser, bare)

    with_selection = parser.parse_args(["-o", str(tmp_path), "--Remove-TMD", "name BB SC1"])
    arg_helper.validate_rotation_args(parser, with_selection)  # does not raise


def test_write_replay_file_round_trips_all_three_remove_tmd_states(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)

    cases = [
        (["-o", str(tmp_path)], False),
        (["-o", str(tmp_path), "-C", "name BB", "--Remove-TMD"], True),
        (["-o", str(tmp_path), "--Remove-TMD", "name BB SC1"], "name BB SC1"),
    ]
    for argv, expected in cases:
        ns = parser.parse_args(argv)
        replay_path = tmp_path / "replay.log"
        arg_helper.write_replay_file(str(replay_path), parser, ns)
        replayed = parser.parse_args(arg_helper.load_replay_args(str(replay_path)))
        assert replayed.remove_tmd == expected


def test_apply_replay_aborts_when_checksum_mismatched_and_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"original bytes")

    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)
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
    arg_helper.add_build_arguments(parser, require_inputs=False)
    ns = parser.parse_args(["-o", str(tmp_path), "-f", str(traj)])
    replay_path = tmp_path / "replay.log"
    arg_helper.write_replay_file(str(replay_path), parser, ns)

    traj.write_bytes(b"changed bytes - different trajectory content now")
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    pre_ns = argparse.Namespace(replay=str(replay_path))
    result = arg_helper.apply_replay(parser, pre_ns, ["-o", str(tmp_path), "-f", str(traj)])
    assert result.trajectory == str(traj)


@pytest.mark.parametrize("token", ["POPC", "POPC:PO4", "TCL1:PO41,PO42"])
def test_lipids_species_token_accepts_valid_formats(token: str) -> None:
    assert arg_helper.lipids_species_token(token) == token  # returned unchanged, for replay round-tripping


@pytest.mark.parametrize("token", ["", ":PO4", "POPC:", "POPC:PO4,", "POPC:PO4,,PO42"])
def test_lipids_species_token_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        arg_helper.lipids_species_token(token)


def test_add_build_arguments_includes_rotation_and_center_by_default() -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False)
    ns = parser.parse_args(["-o", "out"])
    assert hasattr(ns, "center")
    assert hasattr(ns, "rotate")
    assert hasattr(ns, "rotation_direction")


def test_add_build_arguments_can_omit_rotation_and_center() -> None:
    parser = argparse.ArgumentParser()
    arg_helper.add_build_arguments(parser, require_inputs=False, include_rotation=False, include_center=False)
    ns = parser.parse_args(["-o", "out"])
    assert not hasattr(ns, "center")
    assert not hasattr(ns, "rotate")
    assert not hasattr(ns, "rotation_direction")
    with pytest.raises(SystemExit):
        parser.parse_args(["-o", "out", "--center", "protein"])
