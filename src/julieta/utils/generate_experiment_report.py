import base64
import json
import sys
from pathlib import Path

import yaml

_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{id} - {name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; }}
  body {{ color: #1a1a1a; }}
  h1 {{ margin-bottom: 0; }}
  .status {{ color: #555; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  td, th {{ border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: left; }}
  img {{ max-width: 100%; margin: 0.5rem 0 1.5rem; border: 1px solid #ddd; }}
  .run-info {{ color: #555; }}
</style>
</head>
<body>
<h1>{id} - {name}</h1>
<p class="status">Estado: {status}</p>

<h2>Objetivo</h2>
<p>{objective}</p>

<h2>Conclusión</h2>
<p>{conclusion}</p>

<h2>Run de MLflow</h2>
<p class="run-info">run_id: <code>{run_id}</code> — run_name: <code>{run_name}</code></p>

<h2>Métricas</h2>
<table>
<tr><th>Métrica</th><th>Valor</th></tr>
{metrics_rows}
</table>

<h2>Artefactos</h2>
{images_html}
</body>
</html>
"""


def build_report(experiment_dir) -> Path:
    """Arma reports/report.html a partir de metadata.yaml y results/run_summary.json.

    Mismo espíritu que `log_run` (ADR 0009): centralizar el reporte en un solo
    lugar para que cada experimento no lo arme a mano con su propio formato.
    Las imágenes quedan embebidas en base64 para que el archivo sea autocontenido
    (se puede abrir o compartir solo, sin depender de las rutas de `results/`).
    """
    experiment_dir = Path(experiment_dir)

    with open(experiment_dir / "metadata.yaml", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    with open(experiment_dir / "results" / "run_summary.json", encoding="utf-8") as file:
        run_summary = json.load(file)

    results_dir = experiment_dir / "results"
    images_html = ""
    for artifact_name in run_summary.get("artifacts", []):
        image_path = results_dir / artifact_name
        if image_path.suffix.lower() != ".png" or not image_path.exists():
            continue
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        images_html += (
            f"<h3>{artifact_name}</h3>"
            f'<img src="data:image/png;base64,{encoded}" alt="{artifact_name}">\n'
        )

    metrics_rows = "\n".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in run_summary.get("metrics", {}).items()
    )

    html = _TEMPLATE.format(
        id=metadata["id"],
        name=metadata["name"],
        status=metadata["status"],
        objective=metadata.get("objective") or "(sin llenar)",
        conclusion=metadata.get("conclusion") or "(sin llenar)",
        run_id=run_summary.get("run_id", ""),
        run_name=run_summary.get("run_name", ""),
        metrics_rows=metrics_rows,
        images_html=images_html,
    )

    out_path = experiment_dir / "reports" / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    if len(sys.argv) != 2:
        print("uso: julieta-report experiments/IDXX-nombre_del_experimento")
        sys.exit(1)
    out_path = build_report(sys.argv[1])
    print(f"reporte generado en {out_path}")


if __name__ == "__main__":
    main()
