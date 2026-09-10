MASTER_LABEL_MAPPING = {
    "BI RADS 0": -1,
    "BI RADS 1": 0,
    "BI RADS 2": 1,
    "BI RADS 3": 1,
    "BI RADS 4A": 2,
    "BI RADS 4B": 2,
    "BI RADS 4C": 2,
    "BI RADS 5": 2,
    "BI RADS 6": 2,
}
HEALTHY_LABEL_MAPPING = {
    "BI RADS 0": 1,
    "BI RADS 1": 0,
    "BI RADS 2": 0,
    "BI RADS 3": 0,
    "BI RADS 4A": 1,
    "BI RADS 4B": 1,
    "BI RADS 4C": 1,
    "BI RADS 5": 1,
    "BI RADS 6": 1,
    "BI RADS 0": 1,
    "BI RADS 1": 0,
    "BI RADS 2": 0,
    "BI RADS 3": 0,
    "BI RADS 4A": 1,
    "BI RADS 4B": 1,
    "BI RADS 4C": 1,
    "BI RADS 5": 1,
    "BI RADS 6": 1,
}

ANOMALY_LABEL_MAPPING = {
    "BI RADS 0": 1,
    "BI RADS 1": 0,
    "BI RADS 2": 1,
    "BI RADS 3": 1,
    "BI RADS 4A": 1,
    "BI RADS 4B": 1,
    "BI RADS 4C": 1,
    "BI RADS 5": 1,
    "BI RADS 6": 1,
}

BENIGN_ANOMALY_LABEL_MAPPING = {
    "BI RADS 2": 0,
    "BI RADS 3": 1,
    "BI RADS 4A": 1,
    "BI RADS 4B": 1,
    "BI RADS 4C": 1,
    "BI RADS 5": 1,
}


BREAST_DENSITY_LABEL_MAPPING = {
    "tipo a": 0,
    "tipo b": 0,
    "tipo c": 1,
    "tipo d": 1,
    "no": 0,
    "si": 1,
    "tipo A": 0,
    "tipo B": 0,
    "tipo C": 1,
    "tipo D": 1,
    "TIPO A": 0,
    "TIPO B": 0,
    "TIPO C": 1,
    "TIPO D": 1,
}

BRA_CUP_SIZE_LABEL_MAPPING = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "dd": 5}

IRRITATION_OR_REDNESS_PRESENCE_LABEL_MAPPING = {
    "no": 0,
    "si": 1,
    "si izquierdo": 1,
    "si ambos": 1,
    "derecha": 1,
    "izquerda": 1,
    "si derecho": 1,
    "ambas": 1,
}
