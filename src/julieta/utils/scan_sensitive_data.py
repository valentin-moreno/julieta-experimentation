import re
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()

# Nombres de columna/variable que sugieren que se está manejando un dato
# identificable o de salud sin anonimizar. No es infalible (ver docs/
# architecture/data-governance.md): es una segunda capa además de la
# disciplina humana, no un reemplazo.
SENSITIVE_KEYWORD_PATTERNS = [
    re.compile(r"c[eé]dula", re.IGNORECASE),
    re.compile(r"n[uú]mero[_ ]?documento", re.IGNORECASE),
    re.compile(r"documento[_ ]?identidad", re.IGNORECASE),
    re.compile(r"historia[_ ]?cl[ií]nica", re.IGNORECASE),
    re.compile(r"nombre[_ ]?paciente", re.IGNORECASE),
    re.compile(r"diagn[oó]stico", re.IGNORECASE),
    re.compile(r"tarjeta[_ ]?identidad", re.IGNORECASE),
]

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

SCANNED_SUFFIXES = {".py", ".ipynb", ".yaml", ".yml", ".csv", ".json"}

# Marcador para excepciones explícitas y documentadas (ej. el correo de una
# cuenta de servicio, no de una persona). Se pone en la misma línea que el
# falso positivo, con una razón: "... # permitido-datos-sensibles: por qué".
IGNORE_MARKER = "permitido-datos-sensibles"


def find_matches(path: Path):
    """Return a list of (line_number, description) for suspicious content in a file.

    Skips its own source file (which necessarily contains the pattern strings)
    and any file whose extension isn't in SCANNED_SUFFIXES.
    """
    if path.resolve() == THIS_FILE:
        return []
    if path.suffix not in SCANNED_SUFFIXES:
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    matches = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if IGNORE_MARKER in line:
            continue
        for pattern in SENSITIVE_KEYWORD_PATTERNS:
            if pattern.search(line):
                matches.append((line_number, f"palabra clave sensible ({pattern.pattern})"))
        if EMAIL_PATTERN.search(line):
            matches.append((line_number, "posible correo electrónico"))
    return matches


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    had_matches = False
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            continue
        for line_number, description in find_matches(path):
            had_matches = True
            print(f"{path}:{line_number}: {description}")

    if had_matches:
        print(
            "\nSe encontraron posibles datos sensibles. Revisa "
            "docs/architecture/data-governance.md antes de commitear. Si es un falso "
            "positivo, ajusta los patrones en src/julieta/utils/scan_sensitive_data.py."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
