import os
import sys
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import timing, setup_logging
from src.config import RAW_DATA, PROCESSED_FILE

# Configura o logger
setup_logging()
logger = logging.getLogger(__name__)


def combine_csv_files(raw_folder: str, processed_file: str):
    """
    Combina todos os arquivos CSV da pasta especificada em um único arquivo.

    Args:
        raw_folder (str): Caminho para a pasta contendo os arquivos CSV.
        processed_file (str): Caminho para o arquivo combinado a ser salvo.
    """
    logger.info(f"Iniciando a combinação de arquivos CSV na pasta: {raw_folder}")

    # Lista todos os arquivos CSV na pasta raw
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]

    # Verifica se há arquivos CSV na pasta
    if not csv_files:
        logger.error(f"Nenhum arquivo CSV encontrado na pasta: {raw_folder}")
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado na pasta: {raw_folder}")

    logger.info(f"Arquivos CSV encontrados: {csv_files}")

    # Inicializa uma lista para armazenar os DataFrames
    dataframes = []

    # Lê cada arquivo CSV e adiciona ao DataFrame
    for file in csv_files:
        file_path = os.path.join(raw_folder, file)
        logger.debug(f"Lendo o arquivo: {file_path}")
        df = pd.read_csv(file_path, skiprows=1, encoding="utf-8-sig")
        # Renomeia a 8ª coluna para 'Volume'
        if len(df.columns) >= 8:
            df.columns.values[7] = "Volume"
        dataframes.append(df)

    # Combina todos os DataFrames em um único DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(
        f"Arquivos combinados com sucesso. Salvando no arquivo: {processed_file}"
    )

    # Salva o DataFrame combinado no arquivo especificado
    combined_df.to_csv(processed_file, index=False)
    logger.info(f"Arquivo combinado salvo com sucesso: {processed_file}")


@timing
def run_combine_csv_files(raw_folder: str, processed_file: str):
    """
    Executa o processo de combinação de arquivos CSV em um único arquivo processado.

    Args:
        raw_folder (str): Caminho para a pasta contendo os arquivos CSV brutos.
        processed_file (str): Caminho para o arquivo combinado a ser salvo.
    """
    logger.info("Executando a função run_combine_csv_files.")
    # Garante que a pasta processed existe
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    logger.debug(f"Pasta garantida: {os.path.dirname(processed_file)}")
    # Combina os arquivos CSV
    combine_csv_files(raw_folder, processed_file)
    logging.info("Processo de combinação concluído.")


def main():
    """Função principal para ser chamada por outros métodos."""
    try:
        logger.info("Iniciando o processo principal de combinação de CSVs.")
        run_combine_csv_files(RAW_DATA, PROCESSED_FILE)
    except Exception as e:
        logger.exception("Ocorreu um erro inesperado durante a execução.", exc_info=e)


if __name__ == "__main__":
    main()
