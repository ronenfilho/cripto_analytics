import pandas as pd
import numpy as np
from sklearn.base import clone
import matplotlib.pyplot as plt
import os
import sys

import matplotlib.ticker as mticker

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    timing,
    sanitize_symbol,
    get_current_datetime,
    setup_logging,
)
from src.models import walk_forward_prediction
from src.utils import simulate_returns

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)


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

    file_name = f"{get_current_datetime()}_{sanitize_symbol(symbol_to_simulate)}_investment_simulation_{test_period_days}_days.png"

    # Salva o gráfico na pasta figures
    plt.savefig(f"figures/{file_name}", dpi=150, bbox_inches="tight")
    logger.info(f"Gráfico salvo: figures/{file_name}")
