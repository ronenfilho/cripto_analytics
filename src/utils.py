import time
import functools
import pandas as pd

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Iniciando processamento: {func.__name__}...")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Processamento concluído em {elapsed:.2f} segundos.")
            return result
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
            raise
        finally:
            print(f"Finalizando execução: {func.__name__}")
    return wrapper

def filter_symbols(data: pd.DataFrame, symbols: list = None) -> pd.DataFrame:
    """
    Filtra o DataFrame para incluir apenas os símbolos especificados ou lista todos se nenhum filtro for passado.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
        symbols (list, opcional): Lista de símbolos a serem filtrados. Se None, retorna todos os dados.

    Returns:
        pd.DataFrame: DataFrame filtrado contendo apenas os símbolos especificados ou todos os dados.
    """
    if symbols is None:
        return data
    filtered_data = data[data['symbol'].isin(symbols)]
    return filtered_data