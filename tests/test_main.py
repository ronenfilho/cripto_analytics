from src.data_load import main
from unittest.mock import patch
import logging


def test_main(tmp_path):
    # Mock para variáveis globais e funções
    with (
        patch("src.data_load.RAW_DATA", str(tmp_path / "raw")),
        patch(
            "src.data_load.PROCESSED_FILE", str(tmp_path / "processed" / "combined.csv")
        ),
        patch("src.data_load.run_combine_csv_files") as mock_run,
    ):
        # Cria pastas e arquivos temporários
        raw_folder = tmp_path / "raw"
        raw_folder.mkdir()
        processed_file = tmp_path / "processed" / "combined.csv"

        # Executa a função principal
        main()

        # Verifica se a função run_combine_csv_files foi chamada corretamente
        mock_run.assert_called_once_with(str(raw_folder), str(processed_file))


def test_main_with_exception(tmp_path, caplog):
    # Mock para variáveis globais e funções
    with (
        patch("src.data_load.RAW_DATA", str(tmp_path / "raw")),
        patch(
            "src.data_load.PROCESSED_FILE", str(tmp_path / "processed" / "combined.csv")
        ),
        patch(
            "src.data_load.run_combine_csv_files",
            side_effect=Exception("Erro simulado"),
        ) as mock_run,
    ):
        # Executa a função principal e captura logs
        with caplog.at_level(logging.ERROR):
            main()

        # Verifica se a função run_combine_csv_files foi chamada
        mock_run.assert_called_once()

        # Verifica se o log de erro foi gerado
        assert any(
            "Ocorreu um erro inesperado durante a execução." in record.message
            for record in caplog.records
        )
