import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import timing
from src.config import RAW_DATA, PROCESSED_FILE


def combine_csv_files(raw_folder: str, processed_file: str):
    """
    Combina todos os arquivos CSV da pasta especificada em um único arquivo.

    Args:
        raw_folder (str): Caminho para a pasta contendo os arquivos CSV.
        processed_file (str): Caminho para o arquivo combinado a ser salvo.
    """
    # Lista todos os arquivos CSV na pasta raw
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]

    # Verifica se há arquivos CSV na pasta
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado na pasta: {raw_folder}")

    # Inicializa uma lista para armazenar os DataFrames
    dataframes = []

    # Lê cada arquivo CSV e adiciona ao DataFrame
    for file in csv_files:
        file_path = os.path.join(raw_folder, file)
        df = pd.read_csv(file_path, skiprows=1)
        # Renomeia a 8ª coluna para 'Volume'
        if len(df.columns) >= 8:
            df.columns.values[7] = 'Volume'
        dataframes.append(df)

    # Combina todos os DataFrames em um único DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Salva o DataFrame combinado no arquivo especificado
    combined_df.to_csv(processed_file, index=False)

@timing
def run_combine_csv_files(raw_folder, processed_file):
    # Garante que a pasta processed existe
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    # Combina os arquivos CSV
    combine_csv_files(raw_folder, processed_file)

if __name__ == "__main__":
    try:
        run_combine_csv_files(RAW_DATA, PROCESSED_FILE)
    except Exception:
        print("Ocorreu um erro inesperado. Verifique os logs acima.")