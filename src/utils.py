import time
import functools
import pandas as pd
import os
import datetime


# Ajusta o import para garantir que o dotenv seja resolvido corretamente
try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("O pacote 'python-dotenv' não está acessível. Certifique-se de que está instalado corretamente.")

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
USE_TIMING = os.getenv("USE_TIMING", "True").lower() == "true"

# Define o decorator timing
def timing(func):
    """
    Decorator para medir o tempo de execução de uma função.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if USE_TIMING:
            print(f"{func.__name__} - Iniciando processamento...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"{func.__name__} - Processamento concluído em {elapsed:.2f} segundos.")
                return result
            except Exception as e:
                print(f"{func.__name__} - Erro durante o processamento: {e}")
                raise
            finally:
                print(f"{func.__name__} - Finalizando execução.")
        else:
            return func(*args, **kwargs)

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

def sanitize_symbol(symbol: str) -> str:
    """
    Substitui '/' por '_' no nome do símbolo.

    Args:
        symbol (str): Nome do símbolo.

    Returns:
        str: Nome do símbolo sanitizado.
    """
    return symbol.replace('/', '_')

def get_current_datetime() -> str:
    """
    Gera um prefixo com a data e hora atual no formato 'YYYYMMDD_HHMM'.

    Returns:
        str: Prefixo com data e hora atual.
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H%M')