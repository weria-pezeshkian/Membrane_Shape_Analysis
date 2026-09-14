"""Tests for CALM/gui/'s non-Tkinter logic: introspect.py (deriving FieldSpecs from each
command's real argparse parser), path_specs.py (the hand-curated Browse-button table), and
runner.py's build_argv (turning the GUI's current widget values back into a real CALM argv).

widgets.py/app.py (the actual Tkinter widgets) have no coverage here - this environment has no
tkinter installed at all, so they can only be exercised on a machine that has it. Everything
covered here is the pure-Python logic those modules are built on top of.
"""

from __future__ import annotations

from CALM.gui.introspect import MODULES, CommandSpec, field_specs
from CALM.gui.path_specs import path_spec_for
from CALM.gui.runner import build_argv


def _all_commands() -> list[CommandSpec]:
    commands: list[CommandSpec] = []
    for entry in MODULES.values():
        commands.extend([entry] if isinstance(entry, CommandSpec) else entry)
    return commands


def test_modules_covers_every_leaf_command() -> None:
    assert set(MODULES.keys()) == {"Calibrate", "Analyze", "Link", "Map"}
    assert isinstance(MODULES["Calibrate"], CommandSpec)  # the one command with no submodules
    assert [c.label for c in MODULES["Analyze"]] == ["sft", "full", "lipids", "diffusion"]
    assert [c.label for c in MODULES["Link"]] == ["write_ndx", "vmd_xtc", "vmd_vectors"]
    assert [c.label for c in MODULES["Map"]] == [
        "plot", "dynamic_plot", "radial_plot", "lipids_plot", "diffusion_plot",
    ]


def test_every_command_parser_builds_and_yields_field_specs() -> None:
    for command in _all_commands():
        specs = field_specs(command.build_parser())
        assert len(specs) > 0
        assert all(spec.flag.startswith("-") for spec in specs)


def test_field_specs_classifies_bool_optional_action() -> None:
    sft_specs = {s.dest: s for s in field_specs(MODULES["Analyze"][0].build_parser())}
    assert sft_specs["clear"].kind == "bool_optional"


def test_field_specs_classifies_plain_store_true() -> None:
    sft_specs = {s.dest: s for s in field_specs(MODULES["Analyze"][0].build_parser())}
    assert sft_specs["rotate"].kind == "bool"


def test_field_specs_classifies_multichoice_vs_choice() -> None:
    full_specs = {s.dest: s for s in field_specs(MODULES["Analyze"][1].build_parser())}
    assert full_specs["method"].kind == "multichoice"
    assert full_specs["method"].choices == [
        "thickness", "Z_fitted", "mean", "gaussian", "principal", "principal_directions",
    ]

    diffusion_plot_specs = {s.dest: s for s in field_specs(MODULES["Map"][4].build_parser())}
    assert diffusion_plot_specs["scale"].kind == "choice"
    assert diffusion_plot_specs["scale"].choices == ["loglog", "linear"]


def test_field_specs_classifies_multi_without_choices() -> None:
    lipids_specs = {s.dest: s for s in field_specs(MODULES["Analyze"][2].build_parser())}
    assert lipids_specs["lipids"].kind == "multi"
    assert lipids_specs["lipids"].choices is None


def test_field_specs_classifies_remove_tmd_as_optional_flag_value() -> None:
    # --Remove-TMD is nargs='?' with const=True and default=False - a plain
    # "text" classification would str(False) into the Entry's own initial
    # text, so an untouched field would submit the literal string "False"
    # as the flag's value (a real bug this kind exists to prevent).
    sft_specs = {s.dest: s for s in field_specs(MODULES["Analyze"][0].build_parser())}
    assert sft_specs["remove_tmd"].kind == "optional_flag_value"
    assert sft_specs["remove_tmd"].default is False


def test_field_specs_classifies_replica_as_append() -> None:
    # --replica (map/replica_average.py) is action="append" - one value per
    # occurrence (--replica a --replica b), unlike "multi" (--lipids a b) -
    # must not fall through to "multi" or "text".
    radial_specs = {s.dest: s for s in field_specs(MODULES["Map"][2].build_parser())}
    assert radial_specs["replicas"].kind == "append"
    assert radial_specs["replicas"].default == []

    diffusion_plot_specs = {s.dest: s for s in field_specs(MODULES["Map"][4].build_parser())}
    assert diffusion_plot_specs["replicas"].kind == "append"


def test_field_specs_excludes_help_and_man_actions() -> None:
    specs = field_specs(MODULES["Calibrate"].build_parser())
    dests = {s.dest for s in specs}
    assert "help" not in dests
    assert not any(s.flag == "--man" for s in specs)


def test_index_and_index_file_are_both_present_and_never_argparse_required() -> None:
    # Both flags exist (the --index-file split) and neither is hard-required
    # at the raw argparse level - CALM.core.argument_parser.validate_index_arguments
    # enforces the actual either/or requirement post-parse, not argparse itself.
    for command in MODULES["Analyze"]:
        parser = command.build_parser()
        actions = {a.dest: a for a in parser._actions}
        assert actions["index"].required is False
        assert actions["index_file"].required is False


def test_field_spec_required_follows_the_required_arguments_group_not_argparse_required() -> None:
    # -n/--index and --index-file are argparse-optional (see above) but both
    # live in every 'analyze' command's own "Required arguments" argument
    # group, since one of them genuinely is required - FieldSpec.required
    # (what gui/app.py's CollapsibleSection split actually uses) must follow
    # that group, not the raw, individually-misleading argparse flag.
    for command in MODULES["Analyze"]:
        specs = {s.dest: s for s in field_specs(command.build_parser())}
        assert specs["index"].kind == "text"
        assert specs["index_file"].kind == "text"
        assert specs["index"].required is True
        assert specs["index_file"].required is True


def test_field_spec_required_follows_group_for_analyze_full_sft_alternative() -> None:
    # --sft, -f/--trajectory, and -s/--structure are all argparse-optional
    # in 'analyze full' (the real requirement is "--sft, or all of these"),
    # but all three sit in its "Required arguments" group - FieldSpec.required
    # must reflect that, not the individually-misleading argparse flag.
    parser = MODULES["Analyze"][1].build_parser()  # "full"
    actions = {a.dest: a for a in parser._actions}
    assert actions["sft"].required is False
    assert actions["trajectory"].required is False
    assert actions["structure"].required is False

    specs = {s.dest: s for s in field_specs(parser)}
    assert specs["sft"].required is True
    assert specs["trajectory"].required is True
    assert specs["structure"].required is True


def test_field_spec_required_falls_back_to_argparse_required_with_no_custom_groups() -> None:
    # 'link vmd_vectors' has no custom "Required"/"Optional" argument groups
    # at all - FieldSpec.required has nothing but the raw argparse flag to
    # go on there, and must fall back to it rather than defaulting to False.
    vmd_vectors = MODULES["Link"][2]
    specs = {s.dest: s for s in field_specs(vmd_vectors.build_parser())}
    assert specs["input"].required is True
    assert specs["output"].required is True
    assert specs["step"].required is False


def test_path_spec_absent_for_index_but_present_for_index_file() -> None:
    # -n/--index is a plain dynamic-selection string post-split, not a path -
    # it must fall through to a plain text entry with no Browse button.
    assert path_spec_for("sft", "index") is None
    spec = path_spec_for("sft", "index_file")
    assert spec is not None
    assert spec.mode == "open"
    assert spec.extensions == [".ndx"]


def test_path_spec_distinguishes_out_as_directory_vs_file_by_command() -> None:
    # The same dest ("out") means a directory for every 'analyze' command,
    # but a single .ndx file for 'write_ndx' - this only works if the table
    # is scoped by (command, dest), not by dest alone.
    assert path_spec_for("sft", "out").mode == "dir"
    assert path_spec_for("write_ndx", "out").mode == "save"
    assert path_spec_for("write_ndx", "out").extensions == [".ndx"]


def test_path_spec_absent_for_replica_despite_being_directory_valued() -> None:
    # --replica ("append" kind) is deliberately absent from the table, like
    # --lipids ("multi" kind): a Browse button replaces the whole field's
    # text on pick (widgets.py's _browse), which is fine for a single-value
    # field but would erase an already-picked replica instead of adding to
    # it for a multi-value one - no "multi"/"append" field has a path_spec
    # for that reason, so this one falls through to the plain text entry.
    assert path_spec_for("radial_plot", "replicas") is None
    assert path_spec_for("diffusion_plot", "replicas") is None


def test_path_spec_none_for_every_field_with_no_table_entry() -> None:
    for command in _all_commands():
        for spec in field_specs(command.build_parser()):
            path_spec = path_spec_for(command.label, spec.dest)
            if path_spec is not None:
                assert path_spec.mode in ("open", "save", "dir")


def test_build_argv_includes_prefix_and_omits_empty_optional_fields() -> None:
    sft = MODULES["Analyze"][0]
    specs = field_specs(sft.build_parser())
    argv = build_argv(sft.argv_prefix, specs, {
        "trajectory": "t.xtc", "structure": "s.tpr", "index": "name PO4", "out": "outdir",
        "lambda_x": "", "clear": False,
    })
    assert argv[:3] == ["CALM", "analyze", "sft"]
    assert "--trajectory" in argv and "t.xtc" in argv
    assert "--lambda_x" not in argv  # empty string omitted, not passed as --lambda_x ""
    assert "--index-file" not in argv  # not given at all


def test_build_argv_bool_optional_is_always_explicit() -> None:
    sft = MODULES["Analyze"][0]
    specs = field_specs(sft.build_parser())
    base = {"trajectory": "t.xtc", "structure": "s.tpr", "index": "name PO4", "out": "outdir"}

    argv_true = build_argv(sft.argv_prefix, specs, {**base, "clear": True})
    assert "--clear" in argv_true and "--no-clear" not in argv_true

    argv_false = build_argv(sft.argv_prefix, specs, {**base, "clear": False})
    assert "--no-clear" in argv_false and "--clear" not in argv_false


def test_build_argv_plain_bool_omitted_when_false() -> None:
    sft = MODULES["Analyze"][0]
    specs = field_specs(sft.build_parser())
    base = {"trajectory": "t.xtc", "structure": "s.tpr", "index": "name PO4", "out": "outdir"}

    argv_on = build_argv(sft.argv_prefix, specs, {**base, "regularize": True})
    assert "--regularization" in argv_on

    argv_off = build_argv(sft.argv_prefix, specs, {**base, "regularize": False})
    assert "--regularization" not in argv_off


def test_build_argv_multichoice_expands_every_selected_value() -> None:
    full = MODULES["Analyze"][1]
    specs = field_specs(full.build_parser())
    argv = build_argv(full.argv_prefix, specs, {
        "sft": "sftdir", "out": "outdir", "method": ["mean", "thickness"],
    })
    assert "--method" in argv
    i = argv.index("--method")
    assert argv[i + 1:i + 3] == ["mean", "thickness"]


def test_build_argv_append_emits_the_flag_once_per_value() -> None:
    radial = MODULES["Map"][2]
    specs = field_specs(radial.build_parser())
    argv = build_argv(radial.argv_prefix, specs, {
        "numpys_directory": "dir0", "replicas": ["dirA", "dirB"],
    })
    # One occurrence of the flag per value, not one occurrence trailed by
    # both values (that would be "multi"'s shape, wrong for action="append").
    assert argv.count("--replica") == 2
    i = argv.index("--replica")
    assert argv[i + 1] == "dirA"
    j = argv.index("--replica", i + 1)
    assert argv[j + 1] == "dirB"


def test_build_argv_append_omitted_when_empty() -> None:
    radial = MODULES["Map"][2]
    specs = field_specs(radial.build_parser())
    argv = build_argv(radial.argv_prefix, specs, {"numpys_directory": "dir0", "replicas": []})
    assert "--replica" not in argv


def test_build_argv_optional_flag_value_three_states() -> None:
    sft = MODULES["Analyze"][0]
    specs = field_specs(sft.build_parser())
    base = {"trajectory": "t.xtc", "structure": "s.tpr", "index": "name PO4", "out": "outdir"}

    argv_omitted = build_argv(sft.argv_prefix, specs, {**base, "remove_tmd": False})
    assert "--Remove-TMD" not in argv_omitted

    argv_bare = build_argv(sft.argv_prefix, specs, {**base, "remove_tmd": True})
    assert "--Remove-TMD" in argv_bare and "True" not in argv_bare  # bare flag, not "--Remove-TMD True"

    argv_value = build_argv(sft.argv_prefix, specs, {**base, "remove_tmd": "name BB SC1"})
    i = argv_value.index("--Remove-TMD")
    assert argv_value[i + 1] == "name BB SC1"


def test_build_argv_multi_omitted_when_empty() -> None:
    lipids = MODULES["Analyze"][2]
    specs = field_specs(lipids.build_parser())
    argv = build_argv(lipids.argv_prefix, specs, {
        "trajectory": "t.xtc", "structure": "s.tpr", "index": "name PO4", "out": "outdir",
        "lipids": [],
    })
    assert "--lipids" not in argv
