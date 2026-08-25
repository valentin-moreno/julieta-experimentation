import re
from pathlib import Path

from julieta.utils.metadata_schema import ASSIGNED_ID_PATTERN, PENDING_ID

ID_LINE_PATTERN = re.compile(r'^(id:\s*)"' + re.escape(PENDING_ID) + r'"(.*)$', re.MULTILINE)


def _next_available_number(experiments_dir: Path) -> int:
    max_assigned = 0
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "_template":
            continue
        prefix = exp_dir.name.split("-", 1)[0]
        if ASSIGNED_ID_PATTERN.match(prefix):
            max_assigned = max(max_assigned, int(prefix.removeprefix("ID")))
    return max_assigned + 1


def _rewrite_id_in_metadata(metadata_path: Path, new_id: str) -> None:
    text = metadata_path.read_text(encoding="utf-8")
    new_text, count = ID_LINE_PATTERN.subn(rf'\g<1>"{new_id}"\g<2>', text, count=1)
    if count != 1:
        raise ValueError(f'No se encontró una línea id: "{PENDING_ID}" en {metadata_path}')
    metadata_path.write_text(new_text, encoding="utf-8")


def assign_pending_ids(experiments_dir="experiments"):
    """Assign a real sequential ID to every experiments/IDXX-* folder still pending.

    Renames each pending folder and rewrites its metadata.yaml id field in place.
    Returns a list of (old_name, new_name) tuples for everything that was assigned.
    """
    base_path = Path(experiments_dir)
    if not base_path.exists():
        return []

    pending_dirs = sorted(
        d for d in base_path.iterdir() if d.is_dir() and d.name.split("-", 1)[0] == PENDING_ID
    )

    next_number = _next_available_number(base_path)
    assigned = []

    for exp_dir in pending_dirs:
        new_id = f"ID{next_number:02d}"
        suffix = exp_dir.name.split("-", 1)[1] if "-" in exp_dir.name else ""
        new_name = f"{new_id}-{suffix}" if suffix else new_id

        metadata_path = exp_dir / "metadata.yaml"
        if metadata_path.exists():
            _rewrite_id_in_metadata(metadata_path, new_id)

        new_path = exp_dir.with_name(new_name)
        exp_dir.rename(new_path)

        assigned.append((exp_dir.name, new_name))
        next_number += 1

    return assigned


def main():
    assigned = assign_pending_ids()
    if not assigned:
        print("No hay experimentos pendientes de asignación de ID.")
        return

    for old_name, new_name in assigned:
        print(f"{old_name} -> {new_name}")
    print(f"{len(assigned)} experimento(s) asignado(s).")


if __name__ == "__main__":
    main()
