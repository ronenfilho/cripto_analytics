import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging
from src.config import PROCESSED_FILE, SYMBOLS
from src.utils import timing, filter_symbols, setup_logging, compare_variability
from src.plot import (        
    generate_visualizations,
    plot_variability,
    plot_normalized_variability,
    plot_price_trends_by_month_year,
)

# Ignora todos os warnings
warnings.filterwarnings("ignore")


setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)

# Ignora logs do Matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)



@timing
def calculate_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula medidas resumo e de dispersão para os preços de fechamento agrupados por 'symbol'.
    Se um símbolo for informado, retorna apenas para esse símbolo.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com as medidas resumo e de dispersão agrupadas por 'symbol'.
    """
    logger.info("Iniciando a função calculate_summary_statistics.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    grouped_stats = data.groupby("symbol")["close"].describe()
    logger.debug(f"Medidas resumo calculadas: {grouped_stats}")

    logger.info("Função calculate_summary_statistics concluída com sucesso.")
    return grouped_stats



@timing
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features baseadas em séries temporais para previsão de preços.

    Args:
        data (pd.DataFrame): DataFrame com colunas ['date', 'symbol', 'close'].

    Returns:
        pd.DataFrame: DataFrame com novas colunas de features.
    """
    logger.info(
        "Calculando features baseadas em séries temporais para previsão de preços."
    )

    data = data.copy()
    data["return_1d"] = data["close"].pct_change(1)
    data["symbol_original"] = data["symbol"]

    data = data.groupby("symbol", group_keys=False).apply(
        lambda df: df.assign(
            mean_7d=df["close"].rolling(window=7).mean(),
            std_7d=df["close"].rolling(window=7).std(),
            return_7d=df["close"].pct_change(7),
            rolling_max_7d=df["close"].rolling(window=7).max(),
            rolling_min_7d=df["close"].rolling(window=7).min(),
            momentum_7d=df["close"] - df["close"].shift(7),
            volatility_7d=df["return_1d"].rolling(window=7).std(),
        ),
        include_groups=False,
    )

    data = data.rename(columns={"symbol_original": "symbol"})

    expected_cols = ["mean_7d", "std_7d"]
    for col in expected_cols:
        if col not in data.columns:
            logger.error(f"A coluna '{col}' não foi calculada corretamente.")
            raise ValueError(f"A coluna '{col}' não foi calculada corretamente.")

    logger.debug("Features calculadas com sucesso.")
    return data


def main() -> None:
    """Função principal para ser chamada por outros métodos.
    Carrega os dados processados, filtra conforme os símbolos selecionados, calcula estatísticas resumo e gera visualizações.
    """

    data = pd.read_csv(PROCESSED_FILE)

    symbols = SYMBOLS

    logger.info(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Use filter_symbols para filtrar os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula estatísticas resumo e de dispersão para as criptomoedas filtradas
    grouped_stats = calculate_summary_statistics(filtered_data)
    logger.debug(grouped_stats)

    # Gera boxplots e histogramas dos preços de fechamento para os símbolos filtrados
    generate_visualizations(filtered_data)

    # Análise comparativa da variabilidade entre as moedas
    variability = compare_variability(filtered_data)
    logger.debug(variability)

    # Gera e salva o gráfico de variabilidade
    plot_variability(variability)
    logger.info(
        "Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'"
    )


    # Gera e salva o gráfico de variabilidade normalizada
    plot_normalized_variability(filtered_data)
    logger.info(
        "Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'"
    )


    # Gera e salva gráficos de tendências de preço por mês/ano
    plot_price_trends_by_month_year(filtered_data)
    logger.info("Gráficos de tendências de preço por mês/ano salvos na pasta 'figures'")


if __name__ == "__main__":
    main()
