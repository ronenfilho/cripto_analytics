import os
from src.data_load import combine_csv_files, run_combine_csv_files, main
import pandas as pd
import pytest
from unittest.mock import patch
import logging
import runpy
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_combine_csv_files(tmp_path):
    # Cria arquivos CSV temporários
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    processed_file = tmp_path / "processed.csv"

    csv1 = raw_folder / "file1.csv"
    csv2 = raw_folder / "file2.csv"

    csv1.write_text("Introdução\ncol1,col2\n1,2\n3,4", encoding="utf-8-sig")
    csv2.write_text("Introdução\ncol1,col2\n5,6\n7,8", encoding="utf-8-sig")

    # Executa a função
    combine_csv_files(str(raw_folder), str(processed_file))

    # Verifica o resultado
    assert os.path.exists(processed_file)
    df = pd.read_csv(processed_file)
    assert len(df) == 4  # Deve combinar 4 linhas no total


def test_run_combine_csv_files(tmp_path):
    # Configura pastas temporárias
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    processed_file = tmp_path / "processed" / "combined.csv"

    csv1 = raw_folder / "file1.csv"
    csv2 = raw_folder / "file2.csv"

    csv1.write_text("Introdução\ncol1,col2\n1,2\n3,4", encoding="utf-8-sig")
    csv2.write_text("Introdução\ncol1,col2\n5,6\n7,8", encoding="utf-8-sig")

    # Executa a função
    run_combine_csv_files(str(raw_folder), str(processed_file))

    # Verifica o resultado
    assert os.path.exists(processed_file)
    df = pd.read_csv(processed_file)
    assert len(df) == 4  # Deve combinar 4 linhas no total


def test_combine_csv_files_no_csv(tmp_path):
    # Configura pastas temporárias
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    processed_file = tmp_path / "processed.csv"

    # Executa a função e verifica se FileNotFoundError é lançado
    with pytest.raises(FileNotFoundError):
        combine_csv_files(str(raw_folder), str(processed_file))


def test_combine_csv_files_permission_error(tmp_path):
    # Configura pastas temporárias
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    processed_file = tmp_path / "processed.csv"

    csv1 = raw_folder / "file1.csv"
    csv1.write_text("Introdução\ncol1,col2\n1,2\n3,4", encoding="utf-8-sig")

    # Mock para simular erro de permissão
    with patch("pandas.DataFrame.to_csv", side_effect=PermissionError):
        with pytest.raises(PermissionError):
            combine_csv_files(str(raw_folder), str(processed_file))


def test_run_combine_csv_files_exception(tmp_path):
    # Mock para simular exceção na função combine_csv_files
    with patch(
        "src.data_load.combine_csv_files", side_effect=Exception("Erro simulado")
    ):
        raw_folder = tmp_path / "raw"
        processed_file = tmp_path / "processed" / "combined.csv"

        with pytest.raises(Exception):
            run_combine_csv_files(str(raw_folder), str(processed_file))


def test_main_exception_handling(tmp_path, caplog):
    # Mock para simular exceção na função run_combine_csv_files
    with patch(
        "src.data_load.run_combine_csv_files", side_effect=Exception("Erro simulado")
    ):
        with caplog.at_level(logging.ERROR):
            main()

        # Verifica se o log de erro foi gerado
        assert any(
            "Ocorreu um erro inesperado durante a execução." in record.message
            for record in caplog.records
        )


def test_run_main_directly():
    main()


def test_run_as_script():
    runpy.run_module("src.data_load", run_name="__main__")
