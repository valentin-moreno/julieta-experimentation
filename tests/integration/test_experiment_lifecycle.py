import csv
import subprocess

METADATA_TEMPLATE = """\
id: "{id}"
name: "{name}"
author: "integration-test"
start_date: "2026-08-25"
status: "completed"
domain: "test"
tags:
  - "integration"
dataset_version: "v1"
model_type: "dummy"
metrics:
  primary_metric_name: "accuracy"
  target_value: 0.9
  final_value: 0.95
conclusion: "test conclusion"
"""


def _make_experiment(experiments_dir, folder_name, experiment_id, name):
    exp_dir = experiments_dir / folder_name
    exp_dir.mkdir(parents=True)
    (exp_dir / "metadata.yaml").write_text(
        METADATA_TEMPLATE.format(id=experiment_id, name=name), encoding="utf-8"
    )


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{command} failed:\n{result.stdout}\n{result.stderr}"
    return result


def test_full_experiment_lifecycle_via_cli(tmp_path):
    """Exercises julieta-assign-ids -> julieta-validate -> julieta-summary as a
    real sequence of CLI calls (the same order a PR merge + a scientist running
    the tools would trigger), against a shared filesystem state.
    """
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    _make_experiment(experiments_dir, "ID01-existing", "ID01", "existing_experiment")
    _make_experiment(experiments_dir, "IDXX-brand_new", "IDXX", "brand_new_experiment")

    _run(["julieta-assign-ids"], cwd=tmp_path)

    assert (experiments_dir / "ID02-brand_new").is_dir()
    assert not (experiments_dir / "IDXX-brand_new").exists()

    _run(["julieta-validate"], cwd=tmp_path)
    _run(["julieta-summary"], cwd=tmp_path)

    csv_path = tmp_path / "experiments_summary.csv"
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    ids = {row["id"] for row in rows}
    assert ids == {"ID01", "ID02"}

    new_row = next(row for row in rows if row["id"] == "ID02")
    assert new_row["name"] == "brand_new_experiment"
    assert new_row["status"] == "completed"
