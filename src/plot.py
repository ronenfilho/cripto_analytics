import pandas as pd
import numpy as np
from sklearn.base import clone
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import sys

import matplotlib.ticker as mticker

import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    timing,
    sanitize_symbol,
    setup_logging,
    compare_variability,
)
from src.utils import simulate_returns

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)


def plot_scatter_diagram(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str = "figures/scatter_diagram_model.png",
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
    scatter_file_name = "scatter_diagram.png"

    # Salva o gráfico na pasta figures antes de mostrar
    plt.savefig(f"figures/{scatter_file_name}", dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close()


def plot_investment_simulation(
    models: dict,
    predictions: dict,
    X_sim: pd.DataFrame,
    y_sim: pd.Series,
    dates_test_sim: pd.Series,
    y_test_sim: pd.Series,
    min_train_size: int,
    initial_capital: float,
    symbol_to_simulate: str,
    test_period_days: int,
) -> None:
    """
    Cria o gráfico da simulação de investimento.

    Args:
        models (dict): Dicionário com os modelos treinados
        predictions (dict): Previsões pré-calculadas para os modelos
        X_sim (pd.DataFrame): Features para simulação
        y_sim (pd.Series): Target para simulação
        dates_test_sim (pd.Series): Datas do período de teste
        y_test_sim (pd.Series): Valores reais do período de teste
        min_train_size (int): Tamanho mínimo do conjunto de treino
        initial_capital (float): Capital inicial
        symbol_to_simulate (str): Símbolo da criptomoeda
        test_period_days (int): Período de teste em dias
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plota cada modelo usando previsões pré-calculadas
    for name, y_pred_walk_forward in predictions.items():
        sorted_idx = np.argsort(dates_test_sim.values)
        y_pred_walk_forward = np.array(y_pred_walk_forward)[sorted_idx]

        capital_evolution = simulate_returns(
            y_test_sim, y_pred_walk_forward, initial_capital
        )

        ax.plot(dates_test_sim, capital_evolution, label=f"Estratégia {name}")

    # Estratégia "Buy and Hold"
    y_test_values = y_test_sim.values
    hold_evolution = [
        initial_capital * (price / y_test_values[0]) for price in y_test_values
    ]
    ax.plot(
        dates_test_sim, hold_evolution, label="Estratégia Buy and Hold", linestyle="--"
    )

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

    file_name = f"{sanitize_symbol(symbol_to_simulate)}_investment_simulation_{test_period_days}_days.png"

    # Salva o gráfico na pasta figures
    plt.savefig(f"figures/{file_name}", dpi=150, bbox_inches="tight")
    logger.info(f"Gráfico salvo: figures/{file_name}")


@timing
def generate_visualizations(data: pd.DataFrame) -> None:
    """
    Gera visualizações combinadas de histogramas e boxplots dos preços de fechamento agrupados por 'symbol'.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Iniciando a função generate_visualizations.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    symbols = data["symbol"].unique()
    logger.info(f"Símbolos encontrados: {symbols}")

    os.makedirs("figures", exist_ok=True)

    for symbol in symbols:
        logger.info(f"Processando o símbolo: {symbol}")
        symbol_data = data[data["symbol"] == symbol]

        logger.debug(f"Tamanho dos dados para {symbol}: {symbol_data.shape}")

        sanitized_symbol = symbol.replace("/", "_")

        # Subplots: Histograma à esquerda e Boxplot à direita
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histograma
        axes[0].hist(symbol_data["close"], bins=20, alpha=0.7, color="skyblue")
        axes[0].set_title(f"Histograma - {symbol}")
        axes[0].set_xlabel("Preço de Fechamento")
        axes[0].set_ylabel("Frequência")

        # Boxplot
        axes[1].boxplot(symbol_data["close"])
        axes[1].set_title(f"Boxplot - {symbol}")
        axes[1].set_xlabel("Preço de Fechamento")

        # Ajusta layout e salva o gráfico
        plt.tight_layout()
        plt.savefig(f"figures/{sanitized_symbol}_visual_histogram_boxplot.png", dpi=150)
        plt.close()
        logger.info(f"Visualizações combinadas salvas para o símbolo: {symbol}")


@timing
def plot_variability(variability: pd.DataFrame) -> None:
    """
    Cria e salva um gráfico de barras com a análise da variabilidade entre as criptomoedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Iniciando a função plot_variability.")

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.bar(variability["symbol"], variability["variability"], color="skyblue")
    plt.yscale("log")
    plt.title("Variabilidade entre as criptomoedas (Escala Logarítmica)")
    plt.xlabel("Symbol")
    plt.ylabel("Desvio Padrão (Variabilidade)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/variability_analysis_log.png", dpi=150)
    plt.close()
    logger.info(
        "Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'"
    )

    # Gráfico de barras com variabilidade normalizada
    normalized_variability = variability.copy()
    normalized_variability["variability"] /= normalized_variability["variability"].max()

    plt.figure(figsize=(12, 8))
    plt.bar(normalized_variability["symbol"], normalized_variability["variability"])
    plt.title("Análise de Variabilidade Normalizada")
    plt.xlabel("Símbolo")
    plt.ylabel("Variabilidade Normalizada")
    plt.savefig("figures/variability_analysis_normalized.png", dpi=150)
    plt.close()
    logger.info(
        "Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'"
    )

    logger.info("Função plot_variability concluída com sucesso.")


@timing
def plot_normalized_variability(data: pd.DataFrame) -> None:
    """
    Cria e salva um gráfico de barras com a variabilidade normalizada e anotações.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Criando gráfico de variabilidade normalizada entre as criptomoedas.")

    variability = compare_variability(data)
    max_variability = variability["variability"].max()
    variability["normalized_variability"] = (
        variability["variability"] / max_variability * 100
    )

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        variability["symbol"], variability["normalized_variability"], color="skyblue"
    )
    plt.title("Variabilidade Normalizada entre as Criptomoedas")
    plt.xlabel("Symbol")
    plt.ylabel("Variabilidade Normalizada (%)")
    plt.xticks(rotation=45)

    for bar, value in zip(bars, variability["normalized_variability"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("figures/variability_analysis_normalized.png", dpi=150)
    plt.close()
    logger.info(
        "Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'."
    )


@timing
def plot_price_trends_by_month_year(data: pd.DataFrame) -> None:
    """
    Cria e salva gráficos de linha por símbolo agrupados por mês e ano com o preço de fechamento destacando média, mediana e moda.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """

    logger.info("#################################################################")
    logger.info(
        "Construir gráfico de linha com o preço de fechamento destacando a média, mediana e moda ao longo do tempo."
    )
    logger.info("#################################################################")

    # Garante que a pasta 'figures' existe
    os.makedirs("figures", exist_ok=True)
    data["date"] = pd.to_datetime(data["date"])
    data["month_year"] = data["date"].dt.to_period("M")

    symbols = data["symbol"].unique()

    for symbol in symbols:
        symbol_data = data[data["symbol"] == symbol]
        grouped_data = (
            symbol_data.groupby("month_year")["close"]
            .agg(
                [
                    "mean",
                    "median",
                    lambda x: x.mode()[0] if not x.mode().empty else None,
                    "last",
                ]
            )
            .reset_index()
        )
        grouped_data.columns = ["month_year", "mean", "median", "mode", "close"]
        sanitized_symbol = symbol.replace("/", "_")

        plt.figure(figsize=(20, 10))
        plt.plot(
            grouped_data["month_year"].astype(str),
            grouped_data["mean"],
            label="Média",
            color="green",
            linestyle="--",
        )
        plt.plot(
            grouped_data["month_year"].astype(str),
            grouped_data["median"],
            label="Mediana",
            color="orange",
            linestyle="--",
        )
        plt.plot(
            grouped_data["month_year"].astype(str),
            grouped_data["mode"],
            label="Moda",
            color="red",
            linestyle="--",
        )
        plt.plot(
            grouped_data["month_year"].astype(str),
            grouped_data["close"],
            label="Preço de Fechamento",
            color="blue",
        )

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=48))

        plt.title(f"Tendências de Preço de Fechamento por Mês/Ano - {symbol}")
        plt.xlabel("Mês/Ano")
        plt.ylabel("Preço de Fechamento")
        plt.legend()
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"figures/{sanitized_symbol}_price_trends_month_year.png", dpi=150)
        plt.close()
        logger.debug(f"Gráfico de tendências de preço salvo para o símbolo {symbol}.")
