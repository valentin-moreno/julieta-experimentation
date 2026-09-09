from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from julieta.utils.metadata_schema import ExperimentMetadata


def generate_experiments_summary(
    experiments_dir="experiments", output_file="experiments_summary.csv"
):
    data = []
    base_path = Path(experiments_dir)

    # Revisar si la carpeta experiments existe
    if not base_path.exists():
        print(f"Error: No se encontró la carpeta {experiments_dir}")
        return

    # Iterar por cada subcarpeta dentro de experiments
    for exp_dir in base_path.iterdir():
        # Ignorar la plantilla y archivos sueltos
        if exp_dir.name == "_template" or not exp_dir.is_dir():
            continue

        yaml_path = exp_dir / "metadata.yaml"

        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as file:
                try:
                    metadata = yaml.safe_load(file)
                    ExperimentMetadata.model_validate(metadata)

                    # Aplanar el diccionario para que quepa perfecto en una fila de CSV
                    row = {
                        "id": metadata.get("id"),
                        "name": metadata.get("name"),
                        "author": metadata.get("author"),
                        "colaborators": ", ".join(filter(None, metadata.get("colaborators", []))),
                        "status": metadata.get("status"),
                        "type": metadata.get("type"),
                        "domain": metadata.get("domain"),
                        # Convertir la lista de tags en un string separado por comas
                        "tags": ", ".join(filter(None, metadata.get("tags", [])))
                        if metadata.get("tags")
                        else "",
                        "dataset_version": metadata.get("dataset_version"),
                        "model_type": ", ".join(filter(None, metadata.get("model_type", []))),
                        # Extraer los datos anidados de la sección "metrics"
                        "primary_metric_name": metadata.get("metrics", {}).get(
                            "primary_metric_name"
                        ),
                        "target_value": metadata.get("metrics", {}).get("target_value"),
                        "final_value": metadata.get("metrics", {}).get("final_value"),
                        "start_date": metadata.get("start_date"),
                        "end_date": metadata.get("end_date"),
                        "objective": metadata.get("objective"),
                        "conclusion": metadata.get("conclusion"),
                    }
                    data.append(row)
                except yaml.YAMLError as e:
                    print(f"Error leyendo {yaml_path}: {e}")
                except ValidationError as e:
                    print(f"Metadata inválida en {yaml_path}, se omite:\n{e}")

    # Convertir a DataFrame y guardar
    if data:
        df = pd.DataFrame(data)
        # Ordenar alfabéticamente por ID (ID01, ID02...)
        df = df.sort_values(by="id")
        df.to_csv(output_file, index=False)
        print(f" Éxito: CSV generado con {len(df)} experimentos en '{output_file}'")
    else:
        print("No se encontraron experimentos con un metadata.yaml válido para procesar.")


def main():
    generate_experiments_summary()


if __name__ == "__main__":
    main()
