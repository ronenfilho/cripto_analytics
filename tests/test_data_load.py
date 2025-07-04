import os
from src.data_load import combine_csv_files
import pandas as pd


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
