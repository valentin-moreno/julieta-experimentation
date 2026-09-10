"""Cliente minimo para la API interna que expone los datos clinicos
(https://testm.salvahealth.co/api/v1, con Mongo detras). Solo login y lectura --
nada de esto escribe datos.
"""

import requests

LOGIN_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 30

ENDPOINT_PATHS = {
    "categoricals": "/datalake/categorical",  # ojo: la URL real es singular
    "mammography": "/datalake/mammography",
    "tests": "/datalake/tests",
    "patients": "/datalake/patients",
}


def login(base_url: str, api_key: str, email: str, password: str) -> str:
    """Autentica contra la API y devuelve el token."""
    response = requests.post(
        f"{base_url}/users/login",
        json={"email": email, "password": password},
        headers={"key": api_key},
        timeout=LOGIN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    token = response.json()["data"]["token"]
    return token


def get_records(
    base_url: str, token: str, endpoint: str, query_params: str | None = None
) -> list[dict]:
    """Trae todos los registros de un endpoint, opcionalmente filtrados por query_params
    (ej. "companyId=<id>&limit=0" -- el limit=0 es necesario, si no la API pagina).
    """
    if endpoint not in ENDPOINT_PATHS:
        raise ValueError(f"Endpoint desconocido: {endpoint!r}. Válidos: {list(ENDPOINT_PATHS)}")

    url = base_url + ENDPOINT_PATHS[endpoint]
    response = requests.get(
        url,
        headers={"token": token},
        params={"query": query_params} if query_params else None,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("data", [])
