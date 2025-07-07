import pandas as pd
import numpy as np
from sklearn.base import clone
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
    sanitize_symbol,
    get_current_datetime,
    calculate_correlation_coefficients,
    calculate_standard_error_between_mlp_and_best,
    determine_best_equation,
    calculate_standard_error,
    setup_logging,
)
from src.models import walk_forward_prediction, run_training_data
from src.features import calculate_features

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)

"""
Análise de Performance (com K-Fold): Manteremos sua validação K-Fold original para gerar a tabela de RMSE e provar a performance geral dos modelos.

Simulação de Investimento (com Walk-Forward): Para gerar as previsões necessárias para a simulação, usaremos uma técnica chamada Walk-Forward Validation (ou Validação de Janela Expansível). É o método correto para testar estratégias em séries temporais, pois funciona assim:

Treina com dados do dia 1 ao 100, prevê o dia 101.
Treina com dados do dia 1 ao 101, prevê o dia 102.
E assim por diante...
Isso garante que cada previsão seja feita usando apenas dados do passado, simulando perfeitamente um ambiente real.
"""


@timing
def simulate_returns(
    y_test: pd.Series, y_pred: np.ndarray, initial_capital: float = 1000.0
) -> list:
    """Simula os retornos de uma estratégia de investimento."""

    capital_evolution = [initial_capital]
    y_test_values = y_test.values

    # Ajusta o cálculo de capital_evolution para incluir o capital inicial
    daily_returns = np.where(
        y_pred[1:] > y_test_values[:-1], y_test_values[1:] / y_test_values[:-1], 1
    )
    capital_evolution = np.concatenate(
        ([initial_capital], initial_capital * np.cumprod(daily_returns))
    )

    return capital_evolution.tolist()


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

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    strategy_results = []
    bayes_houd_results = []
    daily_returns = []

    for name, model in models.items():
        y_pred_walk_forward = walk_forward_prediction(
            model, X_sim, y_sim, min_train_size
        )
        y_pred_walk_forward = np.array(y_pred_walk_forward)[sorted_idx]

        # Inicializa o capital acumulado para este modelo
        accumulated_capital = initial_capital

        # Calcula os retornos percentuais diários
        for i in range(1, len(y_test_sim)):
            date = dates_test_sim.iloc[i]
            symbol = symbol_to_simulate
            
            # Usa o preço real para calcular o retorno
            actual_return = (y_test_sim.iloc[i] - y_test_sim.iloc[i - 1]) / y_test_sim.iloc[i - 1]

            # Lógica de reinvestimento diário baseada na previsão
            investimento_realizado = y_pred_walk_forward[i] > y_pred_walk_forward[i - 1]
            
            # Valor reinvestido é o saldo do dia anterior
            valor_reinvestido = accumulated_capital
            
            if investimento_realizado:                
                accumulated_capital = accumulated_capital * (1 + actual_return)
            else:
                actual_return = 0.0

            daily_returns.append({
                "date": date,
                "symbol": symbol,
                "model": name,
                "current_price": y_test_sim.iloc[i],
                "prediction": y_pred_walk_forward[i],
                "investment_made": "Yes" if investimento_realizado else "No",                
                "reinvested_value": valor_reinvestido,
                "final_value": accumulated_capital,
                "return": actual_return,
            })

        capital_evolution = simulate_returns(
            y_test_sim, y_pred_walk_forward, initial_capital
        )
        ax.plot(dates_test_sim, capital_evolution, label=f"Estratégia {name}")
        logger.info(f"Capital final com {name}: U${capital_evolution[-1]:.2f}")

        # Adiciona resultados ao DataFrame de estratégia usando o valor final da simulação diária
        strategy_results.append({
            "crypto": symbol_to_simulate,
            "initial_value": initial_capital,
            "final_value": accumulated_capital,
            "strategy": name,
        })

    # Estratégia "Buy and Hold"
    y_test_values = y_test_sim.values
    hold_evolution = [
        initial_capital * (price / y_test_values[0]) for price in y_test_values
    ]
    ax.plot(
        dates_test_sim, hold_evolution, label="Estratégia Buy and Hold", linestyle="--"
    )
    logger.info(f"Capital final com Buy and Hold: U${hold_evolution[-1]:.2f}")

    # Adiciona resultados ao DataFrame de Bayes-Houd
    bayes_houd_results.append({
        "crypto": symbol_to_simulate,
        "initial_value": initial_capital,
        "final_value": hold_evolution[-1],
        "strategy": "Buy and Hold",
    })

    # Salva os retornos diários em um arquivo CSV
    returns_df = pd.DataFrame(daily_returns)    
    returns_file = os.path.join(PROCESSED_DATA, "simulation_results_days.csv")
    returns_df.to_csv(returns_file, index=False, float_format='%.4f')
    print(f"Retornos diários salvos em: {returns_file}")

    # Salva o arquivo combinado
    combined_results = pd.DataFrame(strategy_results + bayes_houd_results)

    # Adiciona colunas de data inicial, data final e quantidade de dias
    start_date = dates_test_sim.iloc[0]
    end_date = dates_test_sim.iloc[-1]
    combined_results["start_date"] = start_date
    combined_results["end_date"] = end_date
    combined_results["days"] = (end_date - start_date).days + 1

    combined_results["return"] = (
        (combined_results["final_value"] - combined_results["initial_value"])
        / combined_results["initial_value"]
    )    

    combined_results = combined_results.round(4)
    output_file = os.path.join(PROCESSED_DATA, "simulation_results_consolidated.csv")
    combined_results.to_csv(output_file, index=False)
    print(f"Resultados combinados salvos em: {output_file}")

    # Configurações do Gráfico
    ax.set_title(
        f"Evolução do Capital - Simulação Walk-Forward ({symbol_to_simulate})",
        fontsize=16,
    )
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Capital Acumulado (U$)", fontsize=12)
    ax.legend(fontsize=10)
    formatter = mticker.FormatStrFormatter("U$%.0f")
    ax.yaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # Garante que a pasta 'figures' existe
    os.makedirs("figures", exist_ok=True)

    file_name = f"{get_current_datetime()}_{sanitize_symbol(symbol_to_simulate)}_investment_simulation_{test_period_days}_days.png"

    # Salva o gráfico na pasta figures antes de mostrar
    plt.savefig(f"figures/{file_name}", dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close()

    # Valida tipos de dados antes de plotar
    if not np.issubdtype(dates_test_sim.dtype, np.datetime64):
        raise ValueError("dates_test_sim deve ser do tipo datetime64.")

    if not np.issubdtype(np.array(capital_evolution).dtype, np.float64):
        raise ValueError("capital_evolution deve conter valores float válidos.")

    if not np.issubdtype(np.array(hold_evolution).dtype, np.float64):
        raise ValueError("hold_evolution deve conter valores float válidos.")

    # Ajusta o tamanho de 'dates_test_sim' para coincidir com 'capital_evolution'
    dates_test_sim = dates_test_sim[: len(capital_evolution)]

    # Ajusta 'capital_evolution' para garantir que tenha a quantidade informada de elementos
    if len(capital_evolution) < len(dates_test_sim):
        missing_elements = len(dates_test_sim) - len(capital_evolution)
        capital_evolution = np.append(
            capital_evolution, [capital_evolution[-1]] * missing_elements
        )

    # Documentação: Diferença entre Buy and Hold e Modelos (MLP, etc.)
    #
    # Estratégia "Buy and Hold":
    # - Assume que o investidor compra o ativo no início do período e mantém até o final.
    # - Não realiza ajustes baseados em previsões.
    # - Evolução do capital é proporcional ao preço do ativo ao longo do tempo.
    #
    # Estratégia com Modelos (MLP, etc.):
    # - Usa previsões feitas pelo modelo para decidir ajustes diários.
    # - Compara o preço previsto com o preço real para simular decisões de compra/venda.
    # - Evolução do capital depende da precisão das previsões.

    logger.info("#################################################################")
    logger.info("Diferença entre Buy and Hold e Modelos (MLP, etc.):")
    logger.info("#################################################################")

    logger.info("Buy and Hold:")
    logger.info(" - Compra no início e mantém até o final.")
    logger.info(" - Evolução proporcional ao preço inicial e final.")

    logger.info("Modelos (MLP, etc.):")
    logger.info(" - Usa previsões para ajustar posições diariamente.")
    logger.info(" - Evolução depende da precisão das previsões.")

    # Adiciona explicação sobre o cálculo do lucro com o modelo

    logger.info("#################################################################")
    logger.info("Computar o lucro obtido com seu modelo:")
    logger.info("#################################################################")

    logger.info("Caso tenha investido U$ 1,000.00 no primeiro dia de operação:")
    logger.info(" - Refazendo investimentos de todo o saldo acumulado diariamente.")
    logger.info(
        " - Apenas se a previsão do valor de fechamento do próximo dia for superior ao do dia atual."
    )


def plot_scatter_diagram(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str = "figures/scatter_diagram.png",
) -> None:
    """
    Gera um diagrama de dispersão para todos os modelos.
    Salva o gráfico na pasta figures.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        y_pred = model_clone.predict(X)
        ax.scatter(y, y_pred, label=name, alpha=0.6)

    ax.set_title("Diagrama de Dispersão - Modelos", fontsize=16)
    ax.set_xlabel("Valores Reais", fontsize=12)
    ax.set_ylabel("Valores Preditos", fontsize=12)
    ax.legend(fontsize=10)

    # Atualiza o nome do arquivo para o scatter diagram
    scatter_file_name = f"{get_current_datetime()}_scatter_diagram.png"

    # Salva o gráfico na pasta figures antes de mostrar
    plt.savefig(f"figures/{scatter_file_name}", dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close()


def save_simulation_results(
    strategy_results: pd.DataFrame, bayes_houd_results: pd.DataFrame, output_dir: str
):
    """
    Salva os resultados da simulação em arquivos separados.

    Args:
        strategy_results (pd.DataFrame): DataFrame com os resultados da estratégia.
        bayes_houd_results (pd.DataFrame): DataFrame com os resultados Bayes-Houd.
        output_dir (str): Diretório onde os arquivos serão salvos.

    Returns:
        None
    """
    # Certifica-se de que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)

    # Define os caminhos dos arquivos
    strategy_file = os.path.join(output_dir, "strategy_results.csv")
    bayes_houd_file = os.path.join(output_dir, "bayes_houd_results.csv")

    # Salva os DataFrames em arquivos CSV
    strategy_results.to_csv(strategy_file, index=False)
    bayes_houd_results.to_csv(bayes_houd_file, index=False)

    print(f"Resultados da estratégia salvos em: {strategy_file}")
    print(f"Resultados Bayes-Houd salvos em: {bayes_houd_file}")

    # Combina os resultados em um único DataFrame
    combined_results = pd.DataFrame(strategy_results + bayes_houd_results)

    # Calcula a coluna de retorno como porcentagem
    combined_results["return"] = (
        (combined_results["final_value"] - combined_results["initial_value"]) / combined_results["initial_value"]
    ) * 100

    # Arredonda os valores para 2 casas decimais
    combined_results = combined_results.round(2)

    # Salva o arquivo combinado na pasta data/processed
    output_file = os.path.join(PROCESSED_DATA, "simulation_results.csv")
    combined_results.to_csv(output_file, index=False)

    print(f"Resultados combinados salvos em: {output_file}")


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
