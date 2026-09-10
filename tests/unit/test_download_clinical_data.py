import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipelines" / "data" / "download"))
import download_clinical_data as dl  # noqa: E402


def test_parse_validation_handles_dict_and_unquoted_string():
    assert dl._parse_validation({"left": 1}) == {"left": 1}
    assert dl._parse_validation("{left: 1, right: 0}") == {"left": 1, "right": 0}
    assert dl._parse_validation("garbage {{{") is None


def test_download_study_saves_raw_records_without_transforming(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path)

    responses = {
        "categoricals": [{"uid": "1", "age": 40}],
        "mammography": [
            {"uid": "1", "mammogramCategory": "bi rads 0"}
        ],  # sin filtrar, se guarda tal cual
        "tests": [{"uid": "1", "validation": {"gaps_detected": True}}],
    }
    monkeypatch.setattr(
        dl.mongo_api,
        "get_records",
        lambda base_url, token, endpoint, query_params=None: responses[endpoint],
    )
    monkeypatch.setattr(dl, "download_signal_file", lambda study, signal: None)

    dl.download_study("sura", token="fake-token", signals=["resistance"])

    categoricals = pd.read_csv(tmp_path / "categoricals_sura.csv")
    assert categoricals["age"].iloc[0] == 40

    mammography = pd.read_csv(tmp_path / "mammography_sura.csv")
    assert mammography["mammogramCategory"].iloc[0] == "bi rads 0"  # no se descarta, esto es raw

    disconnections = pd.read_csv(tmp_path / "disconnections_sura.csv")
    assert bool(disconnections["gaps_detected"].iloc[0]) is True
