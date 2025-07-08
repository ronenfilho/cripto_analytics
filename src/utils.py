import time
import functools
import pandas as pd
import os
import datetime
import sys
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import USE_TIMING, LOG_LEVEL, PROCESSED_DATA


# Configura o logger
logger = logging.getLogger(__name__)

def timing(func: callable) -> callable:
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
                print(
                    f"{func.__name__} - Processamento concluído em {elapsed:.2f} segundos."
                )
                return result
            except Exception as e:
                print(f"{func.__name__} - Erro durante o processamento: {e}")
                raise
            finally:
                print(f"{func.__name__} - Finalizando execução.")
        else:
            return func(*args, **kwargs)

    return wrapper

@timing
def compare_variability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza uma análise comparativa da variabilidade entre as moedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com a variabilidade (desvio padrão) de cada moeda.
    """

    logger.info("#################################################################")
    logger.info(
        "Analisar a variabilidade entre as criptomoedas com base nas medidas de dispersão."
    )
    logger.info("#################################################################")

    logger.info("Iniciando a função compare_variability.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    variability = data.groupby("symbol")["close"].std().reset_index()
    variability.columns = ["symbol", "variability"]
    logger.debug(f"Variabilidade calculada: {variability}")

    logger.info("Função compare_variability concluída com sucesso.")
    return variability



@timing
def simulate_returns(
    y_test: pd.Series, y_pred: np.ndarray, initial_capital: float = 1000.0
) -> list:
    """
    Simula os retornos de uma estratégia de investimento.
    Args:
        y_test (pd.Series): Série temporal contendo os valores reais do ativo financeiro.
        y_pred (np.ndarray): Array contendo as previsões do modelo para os valores do ativo financeiro.
        initial_capital (float, optional): Capital inicial para a simulação. O valor padrão é 1000.0.
    Returns:
        list: Lista contendo a evolução do capital ao longo do tempo, com base nos retornos simulados.
    """

    capital_evolution = [initial_capital]
    y_test_values = y_test.values

    daily_returns = np.where(
        y_pred[1:] > y_test_values[:-1], y_test_values[1:] / y_test_values[:-1], 1
    )
    capital_evolution = np.concatenate(
        ([initial_capital], initial_capital * np.cumprod(daily_returns))
    )

    return capital_evolution.tolist()

def setup_logging():
    # Define o nível de logging a partir do .env
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
    )


def determine_best_equation(
    models: dict, X: pd.DataFrame, y: pd.Series
) -> tuple[str, float]:
    """
    Determina a equação que melhor representa os regressores.
    """
    best_model = None
    best_score = float("-inf")
    for name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        score = model_clone.score(X, y)
        if score > best_score:
            best_score = score
            best_model = name
    return best_model, best_score


def calculate_standard_error(models: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Calcula o erro padrão para todos os modelos.
    """
    errors = {}
    for name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        y_pred = model_clone.predict(X)
        error = np.sqrt(mean_squared_error(y, y_pred))
        errors[name] = error
    return errors


def calculate_standard_error_between_mlp_and_best(
    models: dict, X: pd.DataFrame, y: pd.Series
) -> float:
    """
    Calcula o erro padrão entre o MLP e o melhor regressor.
    """
    mlp_model = models.get("MLPRegressor")
    best_model_name, _ = determine_best_equation(models, X, y)
    best_model = models.get(best_model_name)

    mlp_clone = clone(mlp_model)
    best_clone = clone(best_model)

    mlp_clone.fit(X, y)
    best_clone.fit(X, y)

    y_pred_mlp = mlp_clone.predict(X)
    y_pred_best = best_clone.predict(X)

    if np.array_equal(y_pred_mlp, y_pred_best):
        print(
            "Aviso: As previsões do MLP e do melhor modelo são idênticas, resultando em erro padrão 0.0."
        )
    error = np.sqrt(mean_squared_error(y_pred_mlp, y_pred_best))
    return error


def calculate_correlation_coefficients(
    models: dict, X: pd.DataFrame, y: pd.Series
) -> dict:
    """
    Calcula os coeficientes de correlação para todos os modelos.
    """
    correlations = {}
    for name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        y_pred = model_clone.predict(X)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        correlations[name] = correlation
    return correlations


def filter_symbols(data: pd.DataFrame, symbols: list[str] = None) -> pd.DataFrame:
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
    filtered_data = data[data["symbol"].isin(symbols)]
    return filtered_data


def sanitize_symbol(symbol: str) -> str:
    """
    Substitui '/' por '_' no nome do símbolo.

    Args:
        symbol (str): Nome do símbolo.

    Returns:
        str: Nome do símbolo sanitizado.
    """
    return symbol.replace("/", "_")


def get_current_datetime() -> str:
    """
    Gera um prefixo com a data e hora atual no formato 'YYYYMMDD_HHMM'.

    Returns:
        str: Prefixo com data e hora atual.
    """

    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def delete_simulation_files():
    """Deleta os arquivos de resultados de simulação, se existirem."""
    
    files_to_delete = [
        os.path.join(PROCESSED_DATA, "simulation_results_consolidated.csv"),
        os.path.join(PROCESSED_DATA, "simulation_results_days.csv"),
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Arquivo deletado: {file_path}")
        else:
            print(f"Arquivo não encontrado: {file_path}")

if __name__ == "__main__":
    setup_logging()