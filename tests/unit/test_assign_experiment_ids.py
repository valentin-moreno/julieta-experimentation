from julieta.utils.assign_experiment_ids import assign_pending_ids

METADATA_TEMPLATE = """\
id: "{id}"                     # Ej: "ID01", "ID02"
name: "{name}"
author: "someone"
start_date: "2026-08-24"
status: "planned"
"""


def _make_experiment(base_path, folder_name, experiment_id, name):
    exp_dir = base_path / folder_name
    exp_dir.mkdir()
    (exp_dir / "metadata.yaml").write_text(
        METADATA_TEMPLATE.format(id=experiment_id, name=name), encoding="utf-8"
    )
    return exp_dir


def test_assigns_next_id_to_single_pending_experiment(tmp_path):
    _make_experiment(tmp_path, "ID01-existing", "ID01", "existing")
    _make_experiment(tmp_path, "IDXX-new_idea", "IDXX", "new_idea")

    assigned = assign_pending_ids(tmp_path)

    assert assigned == [("IDXX-new_idea", "ID02-new_idea")]
    assert (tmp_path / "ID02-new_idea").is_dir()
    assert not (tmp_path / "IDXX-new_idea").exists()
    assert 'id: "ID02"' in (tmp_path / "ID02-new_idea" / "metadata.yaml").read_text()


def test_resolves_multiple_pending_experiments_in_one_pass(tmp_path):
    _make_experiment(tmp_path, "ID03-existing", "ID03", "existing")
    _make_experiment(tmp_path, "IDXX-first", "IDXX", "first")
    _make_experiment(tmp_path, "IDXX-second", "IDXX", "second")

    assigned = assign_pending_ids(tmp_path)

    new_ids = {new_name.split("-", 1)[0] for _, new_name in assigned}
    assert new_ids == {"ID04", "ID05"}
    assert len(assigned) == 2


def test_no_pending_experiments_returns_empty_list(tmp_path):
    _make_experiment(tmp_path, "ID01-existing", "ID01", "existing")

    assert assign_pending_ids(tmp_path) == []


def test_ignores_template_folder(tmp_path):
    _make_experiment(tmp_path, "_template", "IDXX", "nombre_del_experimento")

    assert assign_pending_ids(tmp_path) == []


def test_rewrites_readme_heading_when_present(tmp_path):
    exp_dir = _make_experiment(tmp_path, "IDXX-new_idea", "IDXX", "new_idea")
    (exp_dir / "README.md").write_text(
        "# IDXX - new_idea\n\n## Resumen\n\nAlgo.\n", encoding="utf-8"
    )

    assign_pending_ids(tmp_path)

    readme_text = (tmp_path / "ID01-new_idea" / "README.md").read_text()
    assert readme_text.startswith("# ID01 - new_idea")


def test_missing_readme_does_not_fail_assignment(tmp_path):
    _make_experiment(tmp_path, "IDXX-new_idea", "IDXX", "new_idea")

    assigned = assign_pending_ids(tmp_path)

    assert assigned == [("IDXX-new_idea", "ID01-new_idea")]


def test_readme_without_matching_heading_is_left_untouched(tmp_path):
    exp_dir = _make_experiment(tmp_path, "IDXX-new_idea", "IDXX", "new_idea")
    (exp_dir / "README.md").write_text("# Un título distinto\n", encoding="utf-8")

    assign_pending_ids(tmp_path)

    readme_text = (tmp_path / "ID01-new_idea" / "README.md").read_text()
    assert readme_text == "# Un título distinto\n"
