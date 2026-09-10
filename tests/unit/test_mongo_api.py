import pytest

from julieta.data import mongo_api


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def test_login_returns_token(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse({"data": {"token": "fake-token"}})

    monkeypatch.setattr(mongo_api.requests, "post", fake_post)

    token = mongo_api.login(
        base_url="https://example.com", api_key="key123", email="a@b.com", password="secret"
    )

    assert token == "fake-token"
    assert captured["url"] == "https://example.com/users/login"
    assert captured["json"] == {"email": "a@b.com", "password": "secret"}
    assert captured["headers"] == {"key": "key123"}


def test_get_records_hits_correct_endpoint_and_returns_data(monkeypatch):
    captured = {}

    def fake_get(url, headers, params, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return _FakeResponse({"data": [{"uid": "1"}, {"uid": "2"}]})

    monkeypatch.setattr(mongo_api.requests, "get", fake_get)

    records = mongo_api.get_records(
        "https://example.com",
        "tok",
        "categoricals",
        query_params="registeredByCompanyId=abc&limit=0",
    )

    assert records == [{"uid": "1"}, {"uid": "2"}]
    assert captured["url"] == "https://example.com/datalake/categorical"
    assert captured["headers"] == {"token": "tok"}
    assert captured["params"] == {"query": "registeredByCompanyId=abc&limit=0"}


def test_get_records_rejects_unknown_endpoint():
    with pytest.raises(ValueError, match="Endpoint desconocido"):
        mongo_api.get_records("https://example.com", "tok", "not_a_real_endpoint")
