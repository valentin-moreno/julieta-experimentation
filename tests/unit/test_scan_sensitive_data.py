from julieta.utils.scan_sensitive_data import find_matches, main


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_detects_sensitive_keyword_in_scanned_extension(tmp_path):
    path = _write(tmp_path, "notebook_helper.py", "columna = df['numero_documento']\n")

    matches = find_matches(path)

    assert len(matches) == 1
    assert matches[0][0] == 1


def test_detects_email_address(tmp_path):
    path = _write(tmp_path, "config.yaml", "contacto: alguien@ejemplo.com\n")

    matches = find_matches(path)

    assert any("correo" in description for _, description in matches)


def test_ignores_non_scanned_extensions(tmp_path):
    path = _write(tmp_path, "notas.md", "aquí hablamos de historia_clinica en general\n")

    assert find_matches(path) == []


def test_ignore_marker_suppresses_a_documented_false_positive(tmp_path):
    path = _write(
        tmp_path,
        "pipeline.yml",
        'user.email "bot@example.com"  # permitido-datos-sensibles: cuenta de servicio\n',
    )

    assert find_matches(path) == []


def test_clean_file_has_no_matches(tmp_path):
    path = _write(tmp_path, "clean.py", "def suma(a, b):\n    return a + b\n")

    assert find_matches(path) == []


def test_main_returns_nonzero_when_matches_found(tmp_path, capsys):
    path = _write(tmp_path, "leak.py", "cedula = '123456789'\n")

    exit_code = main([str(path)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "palabra clave sensible" in output


def test_main_returns_zero_for_clean_files(tmp_path):
    path = _write(tmp_path, "clean.py", "x = 1\n")

    assert main([str(path)]) == 0
