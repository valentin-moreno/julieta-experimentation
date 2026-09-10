"""Baja a data/raw/ los datos crudos de los estudios pedidos -- categoricos,
diagnostico de mamografia (permitido-datos-sensibles: nombre de concepto de
dominio en un docstring, no un dato real), validacion/desconexion de cada
medicion (de la API interna), y las señales de impedancia que pidas (de Azure
Blob).

No limpia, no filtra, no une nada -- cada quien arma su propio dataset
despues a partir de estos archivos, como le convenga.

    uv run python pipelines/data/download/download_clinical_data.py \\
        --studies sura cafam clinica_de_mama --signals resistance reactance magnitude phase
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import dotenv_values

from julieta.data import mongo_api
from julieta.data.clinical_reference import (
    AZURE_CONTAINER,
    COMPANY_ID_BY_STUDY,
    MONGO_BASE_URL,
    SIGNAL_FILE_PREFIX,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
_ENV = dotenv_values(REPO_ROOT / ".env")


def _parse_validation(value) -> dict | None:
    """El campo `validation` de un test a veces es un dict, a veces un string
    con formato casi-JSON (claves sin comillas, ej. {left: 1, right: 0}).
    Solo se aplana para que quepa en un CSV -- no se decide qué excluir."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        quoted = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', value.strip())
        try:
            return json.loads(quoted)
        except json.JSONDecodeError:
            return None
    return None


def _save_csv(records: list[dict], filename: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(RAW_DIR / filename, index=False)
    print(f"  {filename}: {len(records)} filas")


def download_signal_file(study: str, signal: str) -> None:
    """Baja el .h5 de una señal de un estudio, solo si no está ya local."""
    filename = f"{SIGNAL_FILE_PREFIX[signal]}_{study.upper()}.h5"
    local_path = RAW_DIR / filename
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"  {filename}: ya estaba local")
        return

    client = BlobServiceClient.from_connection_string(_ENV["AZURE_STORAGE_CONNECTION_STRING"])
    blob_client = client.get_blob_client(container=AZURE_CONTAINER, blob=f"raw-data/{filename}")
    local_path.write_bytes(blob_client.download_blob().readall())
    print(f"  {filename}: descargado")


def download_study(study: str, token: str, signals: list[str]) -> None:
    print(f"{study}:")
    company_id = COMPANY_ID_BY_STUDY[study]

    categoricals = mongo_api.get_records(
        MONGO_BASE_URL,
        token,
        "categoricals",
        query_params=f"registeredByCompanyId={company_id}&limit=0",
    )
    _save_csv(categoricals, f"categoricals_{study}.csv")

    diagnosis = mongo_api.get_records(
        MONGO_BASE_URL,
        token,
        "mammography",
        query_params=f"registeredByCompanyId={company_id}&limit=0",
    )
    _save_csv(diagnosis, f"mammography_{study}.csv")

    tests = mongo_api.get_records(
        MONGO_BASE_URL, token, "tests", query_params=f"providerCompanyId={company_id}&limit=0"
    )
    disconnections = [
        {"uid": record["uid"], **parsed}
        for record in tests
        if (parsed := _parse_validation(record.get("validation"))) is not None
    ]
    _save_csv(disconnections, f"disconnections_{study}.csv")

    for signal in signals:
        download_signal_file(study, signal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, choices=list(COMPANY_ID_BY_STUDY))
    parser.add_argument("--signals", nargs="*", default=[], choices=list(SIGNAL_FILE_PREFIX))
    args = parser.parse_args()

    token = mongo_api.login(
        base_url=MONGO_BASE_URL,
        api_key=_ENV["MONGODB_API_KEY"],
        email=_ENV["MONGODB_USERNAME"],
        password=_ENV["MONGODB_PASSWORD"],
    )
    for study in args.studies:
        download_study(study, token, args.signals)


if __name__ == "__main__":
    main()
