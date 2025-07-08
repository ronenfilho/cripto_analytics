import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SYMBOL_TO_SIMULATE, INITIAL_CAPITAL, TEST_PERIOD_DAYS, PROCESSED_DATA
from src.utils import (
    timing,
    filter_symbols,    
    calculate_correlation_coefficients,
    calculate_standard_error_between_mlp_and_best,
    determine_best_equation,
    calculate_standard_error,
    setup_logging,
    simulate_returns,
)
from src.models import walk_forward_prediction, run_training_data
from src.features import calculate_features
from src.plot import plot_scatter_diagram, plot_investment_simulation

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)


@timing
def run_investment_simulation(
    data: pd.DataFrame,
    symbol_to_simulate: str,
    models: dict,
    initial_capital: float = 1000.0,
    test_period_days: int = 30,
) -> None:
    """
    Executa o fluxo completo de simulação de investimento para um ativo e plota o resultado.
    Simula os últimos 'test_period_days' dias.
    """
    logger.info("#" * 60)
    logger.info(f"PARTE 2: SIMULAÇÃO DE INVESTIMENTO PARA {symbol_to_simulate}")
    logger.info("#" * 60)

    single_symbol_data = filter_symbols(data, [symbol_to_simulate])

    # Garante que os dados estão ordenados por data em ordem crescente
    single_symbol_data["date"] = pd.to_datetime(single_symbol_data["date"])
    single_symbol_data = single_symbol_data.sort_values("date", ascending=True)

    data_with_features = calculate_features(single_symbol_data)
    data_processed = data_with_features.dropna()

    features_cols = ["mean_7d", "std_7d", "return_7d", "momentum_7d", "volatility_7d"]
    X_sim = data_processed[features_cols]
    y_sim = data_processed["close"]
    dates_sim = pd.to_datetime(data_processed["date"])

    min_train_size = len(X_sim) - test_period_days
    if min_train_size < 100:
        logger.error(
            "Erro: Não há dados suficientes para uma simulação com o período de teste solicitado."
        )
        return

    y_test_sim = y_sim.iloc[min_train_size:]
    dates_test_sim = dates_sim.iloc[min_train_size:]

    # Ordena por data para garantir ordem cronológica correta
    sorted_idx = np.argsort(dates_test_sim.values)
    dates_test_sim = dates_test_sim.iloc[sorted_idx]
    y_test_sim = y_test_sim.iloc[sorted_idx]

    # Executa a simulação completa
    simulation_results = run_simulation_core(
        models,
        X_sim,
        y_sim,
        dates_test_sim,
        y_test_sim,
        min_train_size,
        initial_capital,
        symbol_to_simulate,
        test_period_days,
    )

    # Plota o gráfico da simulação
    plot_investment_simulation(
        models,
        X_sim,
        y_sim,
        dates_test_sim,
        y_test_sim,
        min_train_size,
        initial_capital,
        symbol_to_simulate,
        test_period_days,
    )

    logger.info("#################################################################")
    logger.info("Diferença entre Buy and Hold e Modelos (MLP, etc.):")
    logger.info("#################################################################")

    logger.info("Buy and Hold:")
    logger.info(" - Compra no início e mantém até o final.")
    logger.info(" - Evolução proporcional ao preço inicial e final.")

    logger.info("Modelos (MLP, etc.):")
    logger.info(" - Usa previsões para ajustar posições diariamente.")
    logger.info(" - Evolução depende da precisão das previsões.")

    logger.info("#################################################################")
    logger.info("Computar o lucro obtido com seu modelo:")
    logger.info("#################################################################")

    logger.info("Caso tenha investido U$ 1,000.00 no primeiro dia de operação:")
    logger.info(" - Refazendo investimentos de todo o saldo acumulado diariamente.")
    logger.info(
        " - Apenas se a previsão do valor de fechamento do próximo dia for superior ao do dia atual."
    )



def run_simulation_core(
    models: dict,
    X_sim: pd.DataFrame,
    y_sim: pd.Series,
    dates_test_sim: pd.Series,
    y_test_sim: pd.Series,
    min_train_size: int,
    initial_capital: float,
    symbol_to_simulate: str,
    test_period_days: int,
) -> dict:
    """
    Executa o core da simulação de investimento para todos os modelos.
    
    Args:
        models (dict): Dicionário com os modelos treinados
        X_sim (pd.DataFrame): Features para simulação
        y_sim (pd.Series): Target para simulação
        dates_test_sim (pd.Series): Datas do período de teste
        y_test_sim (pd.Series): Valores reais do período de teste
        min_train_size (int): Tamanho mínimo do conjunto de treino
        initial_capital (float): Capital inicial
        symbol_to_simulate (str): Símbolo da criptomoeda
        test_period_days (int): Período de teste em dias
    
    Returns:
        dict: Resultados da simulação incluindo dados diários e consolidados
    """
    models_evolution_data = {}
    strategy_results = []
    daily_returns = []
    

    # Calcula previsões e evolução do capital para cada modelo
    predictions = {}
    for name, model in models.items():
        y_pred_walk_forward = walk_forward_prediction(
            model, X_sim, y_sim, min_train_size
        )
        sorted_idx = np.argsort(dates_test_sim.values)
        y_pred_walk_forward = np.array(y_pred_walk_forward)[sorted_idx]
        predictions[name] = y_pred_walk_forward

        capital_evolution = simulate_returns(
            y_test_sim, y_pred_walk_forward, initial_capital
        )

    # Retorna previsões junto com os resultados
    return {
        "predictions": predictions,
        "strategy_results": strategy_results,
        "daily_returns": daily_returns,
    }

def main(data: pd.DataFrame = None, models: dict = None) -> None:
    """Função principal para executar o fluxo de simulação de investimento e análise de modelos."""

    # --- PARTE 1: Treinamento e Validação dos Modelos ---
    if models is not None and data is not None:
        logger.info("Usando modelos e dados fornecidos pelo usuário.")
    else:
        models, data = run_training_data()

    # Define X e y para análise
    X = data[["mean_7d", "std_7d", "return_7d", "momentum_7d", "volatility_7d"]]
    y = data["close"]

    # --- PARTE 2: Chamada para a Simulação de Investimento ---

    run_investment_simulation(
        data=data,
        symbol_to_simulate=SYMBOL_TO_SIMULATE,
        models=models,
        initial_capital=INITIAL_CAPITAL,
        test_period_days=TEST_PERIOD_DAYS,
    )

    # --- PARTE 3: Análise dos Modelos ---

    logger.info("#################################################################")
    logger.info("PARTE 3: Análise dos Modelos:")
    logger.info("#################################################################")

    # Gera o diagrama de dispersão
    plot_scatter_diagram(models, X, y)

    # Calcula os coeficientes de correlação
    correlations = calculate_correlation_coefficients(models, X, y)
    logger.info("Coeficientes de Correlação:")
    for name, corr in correlations.items():
        logger.info(f" - {name}: {corr:.4f}")

    # Determina a melhor equação
    best_model_name, best_score = determine_best_equation(models, X, y)
    logger.info(f"Melhor Modelo: {best_model_name} com score {best_score:.4f}")

    # Calcula o erro padrão
    errors = calculate_standard_error(models, X, y)
    logger.info("Erro Padrão:")
    for name, error in errors.items():
        logger.info(f" - {name}: {error:.4f}")

    # Calcula o erro padrão entre MLP e o melhor regressor
    error_between_mlp_and_best = calculate_standard_error_between_mlp_and_best(
        models, X, y
    )
    logger.info(
        f"Erro Padrão entre MLP e {best_model_name}: {error_between_mlp_and_best:.4f}"
    )


if __name__ == "__main__":
    main()
