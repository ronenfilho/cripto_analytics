import pandas as pd
import numpy as np

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SYMBOLS_TO_SIMULATE,
    INITIAL_CAPITAL,
    TEST_PERIOD_DAYS,
    PROCESSED_DATA,
)
from src.utils import (
    timing,
    filter_symbols,
    calculate_correlation_coefficients,
    calculate_standard_error_between_mlp_and_best,
    determine_best_equation,
    calculate_standard_error,
    setup_logging,
    simulate_returns,
    delete_simulation_files,
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
    if min_train_size < 30:
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
        simulation_results["predictions"],  # Adicionado o argumento predictions
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

        # Armazena dados para uso posterior
        models_evolution_data[name] = {
            "capital_evolution": capital_evolution,
            "predictions": y_pred_walk_forward,
        }

        # Calcula retornos diários para este modelo
        y_test_values = y_test_sim.values
        current_capital = initial_capital

        for i in range(len(y_test_values) - 1):
            date = dates_test_sim.iloc[i]
            current_price = y_test_values[i]
            next_price = y_test_values[i + 1]
            predicted_price = y_pred_walk_forward[i]

            # Verifica se deve investir (previsão > preço atual)
            investment_made = "Yes" if predicted_price > current_price else "No"

            # Calcula retorno baseado na decisão de investir
            if investment_made == "Yes":
                return_pct = (next_price - current_price) / current_price
                capital_before = current_capital
                capital_invested = capital_before
                capital_after = capital_before * (1 + return_pct)
                capital_gained = capital_after - capital_before
                current_capital = capital_after  # Atualiza o capital para o próximo dia
            else:
                return_pct = 0.0  # Sem investimento, sem retorno
                capital_before = current_capital
                capital_invested = 0.0
                capital_after = current_capital  # Capital permanece igual
                capital_gained = 0.0
                # current_capital permanece igual

            daily_returns.append(
                {
                    "date": date,
                    "symbol": symbol_to_simulate,
                    "strategy": name,
                    "current_price": round(current_price, 2),
                    "predicted_price": round(predicted_price, 2),
                    "next_price": round(next_price, 2),
                    "investment_made": investment_made,
                    "capital_before": round(capital_before, 2),
                    "capital_invested": round(capital_invested, 2),
                    "capital_after": round(capital_after, 2),
                    "capital_gained": round(capital_gained, 2),
                    "return_pct": round(return_pct, 4),
                }
            )

        # Adiciona aos resultados da estratégia
        strategy_results.append(
            {
                "symbol": symbol_to_simulate,
                "strategy": name,
                "initial_value": initial_capital,
                "final_value": round(current_capital, 2),
                "return_pct": round(
                    (current_capital - initial_capital) / initial_capital, 4
                ),
            }
        )

        logger.info(f"Capital final com {name}: U${current_capital:.2f}")

    # Estratégia "Buy and Hold"
    y_test_values = y_test_sim.values
    hold_evolution = [
        initial_capital * (price / y_test_values[0]) for price in y_test_values
    ]

    # Adiciona dados diários para Buy and Hold
    for i in range(len(y_test_values) - 1):
        date = dates_test_sim.iloc[i]
        current_price = y_test_values[i]
        next_price = y_test_values[i + 1]

        return_pct = (next_price - current_price) / current_price
        capital_before = hold_evolution[i]
        capital_after = hold_evolution[i + 1]
        capital_invested = capital_before
        capital_gained = capital_after - capital_before

        daily_returns.append(
            {
                "date": date,
                "symbol": symbol_to_simulate,
                "strategy": "Buy and Hold",
                "current_price": round(current_price, 2),
                "predicted_price": round(
                    current_price, 2
                ),  # Buy and Hold não faz previsão
                "next_price": round(next_price, 2),
                "investment_made": "Yes",  # Buy and Hold sempre investe
                "capital_before": round(capital_before, 2),
                "capital_invested": round(capital_invested, 2),
                "capital_after": round(capital_after, 2),
                "capital_gained": round(capital_gained, 2),
                "return_pct": round(return_pct, 4),
            }
        )

    # Adiciona Buy and Hold aos resultados
    bayes_houd_results = [
        {
            "symbol": symbol_to_simulate,
            "strategy": "Buy and Hold",
            "initial_value": initial_capital,
            "final_value": round(hold_evolution[-1], 2),
            "return_pct": round(
                (hold_evolution[-1] - initial_capital) / initial_capital, 4
            ),
        }
    ]

    logger.info(f"Capital final com Buy and Hold: U${hold_evolution[-1]:.2f}")

    # Certifica-se de que os DataFrames estão definidos antes de salvar
    returns_df = pd.DataFrame(daily_returns)
    combined_results = pd.DataFrame(strategy_results + bayes_houd_results)

    # Salva os retornos diários em um arquivo CSV no modo append
    returns_df["symbol"] = symbol_to_simulate  # Adiciona a coluna do símbolo
    returns_file = os.path.join(PROCESSED_DATA, "simulation_results_days.csv")
    if not os.path.exists(returns_file):
        returns_df.to_csv(returns_file, index=False)
    else:
        returns_df.to_csv(returns_file, mode='a', header=False, index=False)
    print(f"Retornos diários salvos em: {returns_file}")

    # Salva o arquivo combinado no modo append
    combined_results["symbol"] = symbol_to_simulate  # Adiciona a coluna do símbolo
    output_file = os.path.join(PROCESSED_DATA, "simulation_results_consolidated.csv")
    if not os.path.exists(output_file):
        combined_results.to_csv(output_file, index=False)
    else:
        combined_results.to_csv(output_file, mode='a', header=False, index=False)
    print(f"Resultados combinados salvos em: {output_file}")

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

    # Deleta arquivos de simulação anteriores
    delete_simulation_files()

    # Itera sobre cada símbolo para realizar a simulação
    for symbol in SYMBOLS_TO_SIMULATE:
        logger.info(f"Iniciando simulação para {symbol}")
        filtered_data = filter_symbols(data, [symbol])

        run_investment_simulation(
            data=filtered_data,
            symbol_to_simulate=symbol,
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
