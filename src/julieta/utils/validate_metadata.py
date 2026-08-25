import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from julieta.utils.metadata_schema import ExperimentMetadata


def validate_experiments(experiments_dir="experiments"):
    """Validate every experiments/*/metadata.yaml against ExperimentMetadata.

    Returns a list of (path, error) tuples for anything that failed to validate.
    """
    base_path = Path(experiments_dir)
    errors = []

    if not base_path.exists():
        return [(str(base_path), "No se encontró la carpeta de experimentos")]

    for exp_dir in sorted(base_path.iterdir()):
        if exp_dir.name == "_template" or not exp_dir.is_dir():
            continue

        yaml_path = exp_dir / "metadata.yaml"
        if not yaml_path.exists():
            errors.append((str(yaml_path), "No existe metadata.yaml"))
            continue

        with open(yaml_path, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        try:
            metadata = ExperimentMetadata.model_validate(raw)
        except ValidationError as exc:
            errors.append((str(yaml_path), str(exc)))
            continue

        folder_prefix = exp_dir.name.split("-", 1)[0]
        if folder_prefix != metadata.id:
            errors.append(
                (
                    str(yaml_path),
                    f"El id del metadata ({metadata.id!r}) no coincide con el prefijo "
                    f"de la carpeta ({folder_prefix!r})",
                )
            )

    return errors


def main():
    errors = validate_experiments()
    if errors:
        for path, error in errors:
            print(f"[INVALIDO] {path}\n{error}\n")
        print(f"{len(errors)} archivo(s) de metadata inválidos.")
        sys.exit(1)

    print("Todos los metadata.yaml son válidos.")


if __name__ == "__main__":
    main()
