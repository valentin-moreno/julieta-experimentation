import json

from julieta.utils.generate_experiment_report import build_report

METADATA = """\
id: "ID02"
name: "mlops_pipeline_smoke_test"
author: "dgrajales"
start_date: "2026-09-08"
status: "completed"
type: "experiment"
domain: "platform_validation"
dataset_version: "synthetic_v1"
model_type: "xgboost_classifier"
metrics:
  primary_metric_name: "roc_auc"
  target_value: 0.85
  final_value: 0.94
objective: "Validar el pipeline end-to-end."
conclusion: "Funciona de punta a punta."
"""


def _make_experiment(base_path, run_summary):
    exp_dir = base_path / "ID02-mlops_pipeline_smoke_test"
    (exp_dir / "results").mkdir(parents=True)
    (exp_dir / "metadata.yaml").write_text(METADATA, encoding="utf-8")
    (exp_dir / "results" / "run_summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
    for artifact_name in run_summary.get("artifacts", []):
        (exp_dir / "results" / artifact_name).write_bytes(b"fake-png-bytes")
    return exp_dir


def test_build_report_embeds_metrics_and_images(tmp_path):
    exp_dir = _make_experiment(
        tmp_path,
        {
            "run_id": "abc123",
            "run_name": "baseline",
            "metrics": {"test_roc_auc": 0.94, "test_accuracy": 0.87},
            "artifacts": ["confusion_matrix_test.png"],
        },
    )

    out_path = build_report(exp_dir)

    assert out_path == exp_dir / "reports" / "report.html"
    html = out_path.read_text(encoding="utf-8")
    assert "ID02 - mlops_pipeline_smoke_test" in html
    assert "abc123" in html
    assert "baseline" in html
    assert "test_roc_auc" in html and "0.94" in html
    assert "confusion_matrix_test.png" in html
    assert "data:image/png;base64," in html


def test_build_report_handles_missing_objective_and_conclusion(tmp_path):
    metadata_without_narrative = METADATA.replace(
        'objective: "Validar el pipeline end-to-end."\n', ""
    ).replace('conclusion: "Funciona de punta a punta."\n', "")
    exp_dir = tmp_path / "ID02-mlops_pipeline_smoke_test"
    (exp_dir / "results").mkdir(parents=True)
    (exp_dir / "metadata.yaml").write_text(metadata_without_narrative, encoding="utf-8")
    (exp_dir / "results" / "run_summary.json").write_text(
        json.dumps({"run_id": "abc123", "run_name": "baseline", "metrics": {}, "artifacts": []}),
        encoding="utf-8",
    )

    out_path = build_report(exp_dir)

    html = out_path.read_text(encoding="utf-8")
    assert "(sin llenar)" in html
