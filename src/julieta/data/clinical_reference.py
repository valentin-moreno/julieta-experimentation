"""Datos de referencia para traer información clínica de los estudios --
IDs internos y nombres de archivo, no lógica. Ver pipelines/data/download.
"""

MONGO_BASE_URL = "https://testm.salvahealth.co/api/v1"
AZURE_CONTAINER = "ai-julieta"

# IDs internos de la compañía en Mongo -- no son datos de paciente, son el
# identificador de la clínica/estudio en el sistema.
COMPANY_ID_BY_STUDY = {
    "sura": "6759de5f71338585c4848592",
    "cafam": "67fd0d81632852f01f330e91",
    "clinica_de_mama": "67c066596ef94cf42e303220",
    # "santafe": "6840ecaf4240806519f4ad92",  # sin mamografia (0 registros, confirmado 2026-09-09)
}

# Prefijo del nombre de archivo .h5 por señal -- el resto del nombre es
# siempre "_<ESTUDIO EN MAYÚSCULA>.h5", confirmado listando el container.
SIGNAL_FILE_PREFIX = {
    "resistance": "resistance",
    "reactance": "reactance",
    "magnitude": "impedance_magnitude",
    "phase": "impedance_phase",
}
