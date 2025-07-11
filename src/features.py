import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
from src.config import PROCESSED_FILE, SYMBOLS, PROCESSED_DATA
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


def calculate_summary_statistics(
    data: pd.DataFrame,
    output_filename: str = "summary_statistics.csv",
    symbols: list = None,
) -> pd.DataFrame:
    """Calcula medidas resumo e salva em um arquivo CSV.

    Args:
        data (pd.DataFrame): DataFrame contendo as colunas ['symbol', 'close'].
        output_filename (str): Nome do arquivo CSV para salvar os resultados.
        symbols (list, optional): Lista de símbolos para filtrar. Defaults to None (todos os símbolos).

    Returns:
        pd.DataFrame: DataFrame contendo as medidas resumo calculadas
    """
    logger.info("Iniciando a função calculate_summary_statistics.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    # Filtrar por símbolos se fornecidos
    if symbols:
        data = data[data['symbol'].isin(symbols)]
        logger.info(f"Dados filtrados para os símbolos: {symbols}")

    summary = []

    for symbol, group in data.groupby("symbol"):
        close_prices = group["close"].dropna()

        # Formatando valores para ter 2 casas decimais
        mean_value = round(close_prices.mean(), 2)
        median_value = round(close_prices.median(), 2)
        mode_value = (
            round(close_prices.mode().iloc[0], 2)
            if not close_prices.mode().empty
            else None
        )
        min_value = round(close_prices.min(), 2)
        max_value = round(close_prices.max(), 2)

        summary.append(
            {
                "symbol": symbol,
                "mean": mean_value,
                "median": median_value,
                "mode": mode_value,
                "min": min_value,
                "max": max_value,
            }
        )

    # Salvar os resultados em um arquivo CSV
    filepath = PROCESSED_DATA / output_filename

    # Criar DataFrame e formatar colunas numéricas para 2 casas decimais
    df_summary = pd.DataFrame(summary)

    # Garante que todas as colunas numéricas tenham 2 casas decimais no arquivo CSV
    df_summary.to_csv(filepath, index=False, float_format='%.2f')

    logging.info(f"Medidas resumo salvas em: {filepath}")

    return df_summary


def calculate_dispersion_measures(
    data: pd.DataFrame,
    output_filename: str = "dispersion_statistics.csv",
    symbols: list = None,
) -> None:
    """Calcula medidas de dispersão e salva em um arquivo CSV.

    Args:
        data (pd.DataFrame): DataFrame contendo as colunas ['symbol', 'close'].
        output_filename (str): Nome do arquivo CSV para salvar os resultados.
        symbols (list, optional): Lista de símbolos para filtrar. Defaults to None (todos os símbolos).

    Returns:
        None
    """
    logger.info("Iniciando a função calculate_dispersion_measures.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    # Filtrar por símbolos se fornecidos
    if symbols:
        data = data[data['symbol'].isin(symbols)]
        logger.info(f"Dados filtrados para os símbolos: {symbols}")

    dispersion_stats = []

    for symbol, group in data.groupby("symbol"):
        close_prices = group["close"].dropna()

        # Calcula medidas de dispersão
        std_dev = round(close_prices.std(), 2)
        variance = round(close_prices.var(), 2)

        # Calcula o coeficiente de variação (CV = desvio padrão / média * 100)
        mean = close_prices.mean()
        cv = round((std_dev / mean * 100), 2) if mean != 0 else None

        # Amplitude (Range)
        price_range = round(close_prices.max() - close_prices.min(), 2)

        # Quartis
        q1 = round(close_prices.quantile(0.25), 2)
        q3 = round(close_prices.quantile(0.75), 2)

        # IQR (Intervalo Interquartil)
        iqr = round(q3 - q1, 2)

        dispersion_stats.append(
            {
                "symbol": symbol,
                "std_dev": std_dev,
                "variance": variance,
                "coef_variation": cv,
                "range": price_range,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
            }
        )

    # Salvar os resultados em um arquivo CSV
    filepath = PROCESSED_DATA / output_filename

    # Garante que todas as colunas numéricas tenham 2 casas decimais no arquivo CSV
    pd.DataFrame(dispersion_stats).to_csv(filepath, index=False, float_format='%.2f')

    logging.info(f"Medidas de dispersão salvas em: {filepath}")


def analisar_medidas_dispersao(
    data: pd.DataFrame,
    output_filename: str = "analise_dispersao.csv",
    symbols: list = None,
) -> pd.DataFrame:
    """Calcula e compara várias medidas de dispersão entre as criptomoedas.

    Esta função realiza uma análise detalhada das medidas de dispersão para cada criptomoeda,
    permitindo comparar a volatilidade e estabilidade entre diferentes ativos.

    Args:
        data (pd.DataFrame): DataFrame contendo as colunas ['symbol', 'close'].
        output_filename (str): Nome do arquivo CSV para salvar os resultados.
        symbols (list, optional): Lista de símbolos para filtrar. Defaults to None (todos os símbolos).

    Returns:
        pd.DataFrame: DataFrame com as medidas de dispersão comparadas.
    """
    logger.info(
        "Iniciando análise comparativa detalhada das medidas de dispersão entre criptomoedas"
    )
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")

    # Filtrar por símbolos se fornecidos
    if symbols:
        data = data[data['symbol'].isin(symbols)]
        logger.info(f"Dados filtrados para os símbolos: {symbols}")

    # DataFrame para armazenar os resultados
    results = []

    # Agrupar por símbolo
    for symbol, group in data.groupby('symbol'):
        close_prices = group['close'].dropna()

        # Cálculo das medidas de dispersão
        std_dev = round(close_prices.std(), 2)
        variance = round(close_prices.var(), 2)
        price_range = round(close_prices.max() - close_prices.min(), 2)
        mean = round(close_prices.mean(), 2)

        # Coeficiente de variação (CV = desvio padrão / média * 100)
        cv = round((std_dev / mean * 100), 2) if mean != 0 else None

        # Quartis e medidas baseadas em quartis
        q1 = round(close_prices.quantile(0.25), 2)
        q3 = round(close_prices.quantile(0.75), 2)
        iqr = round(q3 - q1, 2)  # Amplitude interquartil

        # Amplitude semi-interquartil (medida de dispersão robusta)
        semi_iqr = round(iqr / 2, 2)

        # Desvio médio absoluto (outra medida robusta)
        # Calculando MAD manualmente, pois Series não tem método .mad()
        mad = round(np.abs(close_prices - close_prices.mean()).mean(), 2)

        # Adicionar resultados
        results.append(
            {
                'symbol': symbol,
                'desvio_padrao': std_dev,
                'variancia': variance,
                'coef_variacao': cv,
                'amplitude': price_range,
                'amplitude_interquartil': iqr,
                'semi_iqr': semi_iqr,
                'desvio_medio_absoluto': mad,
                'media': mean,
                'q1': q1,
                'q3': q3,
            }
        )

    # Criar DataFrame com os resultados
    df_results = pd.DataFrame(results)

    # Ordenar por coeficiente de variação (do mais volátil para o menos volátil)
    df_results = df_results.sort_values(by='coef_variacao', ascending=False)

    # Salvar os resultados em um arquivo CSV
    filepath = PROCESSED_DATA / output_filename
    df_results.to_csv(filepath, index=False, float_format='%.2f')
    logger.info(f"Análise de dispersão salva em: {filepath}")

    # Classificar as criptomoedas com base na volatilidade
    logger.info(
        "Classificação das criptomoedas por volatilidade (coeficiente de variação):"
    )
    for i, row in df_results.iterrows():
        logger.info(
            f"{i+1}. {row['symbol']}: CV = {row['coef_variacao']}%, Desvio Padrão = {row['desvio_padrao']}"
        )

    return df_results


def visualizar_comparacao_dispersao(
    df_dispersao: pd.DataFrame, dados_originais: pd.DataFrame
) -> None:
    """Gera visualizações comparativas das medidas de dispersão entre criptomoedas.

    Args:
        df_dispersao (pd.DataFrame): DataFrame contendo as medidas de dispersão calculadas.
        dados_originais (pd.DataFrame): DataFrame original contendo os preços das criptomoedas.
    """
    logger.info("Gerando visualizações comparativas das medidas de dispersão")

    # Criar pasta para figuras se não existir
    os.makedirs("figures", exist_ok=True)

    # Definir estilo dos gráficos
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Gráfico de barras comparando os coeficientes de variação
    plt.figure(figsize=(14, 8))
    plt.bar(df_dispersao['symbol'], df_dispersao['coef_variacao'], color='skyblue')
    plt.title('Comparação de Volatilidade entre Criptomoedas', fontsize=16)
    plt.xlabel('Criptomoeda', fontsize=12)
    plt.ylabel('Coeficiente de Variação (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for i, v in enumerate(df_dispersao['coef_variacao']):
        plt.text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/comparacao_volatilidade.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Gráfico de radar para comparação multidimensional
    # Selecionar as métricas para o radar
    metrics = [
        'desvio_padrao',
        'variancia',
        'coef_variacao',
        'amplitude_interquartil',
        'desvio_medio_absoluto',
    ]
    metrics_labels = ['Desvio Padrão', 'Variância', 'Coef. Variação', 'IQR', 'MAD']

    # Normalizar os valores para o gráfico de radar
    df_radar = df_dispersao[['symbol'] + metrics].copy()
    for col in metrics:
        max_val = df_radar[col].max()
        if max_val > 0:  # Evitar divisão por zero
            df_radar[col] = df_radar[col] / max_val

    # Criar gráfico de radar
    num_symbols = len(df_dispersao)
    num_metrics = len(metrics)

    # Limitar a 6 criptomoedas para evitar confusão visual
    if num_symbols > 6:
        logger.info("Limitando visualização radar para as 6 criptomoedas mais voláteis")
        df_radar = df_radar.head(6)
        num_symbols = 6

    # Ângulos para o radar
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Fechar o círculo

    # Preparar figura com subplots (um radar por criptomoeda)
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': 'polar'})

    # Cores para cada criptomoeda
    colors = plt.cm.tab10(np.linspace(0, 1, num_symbols))

    for i, (_, row) in enumerate(df_radar.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]  # Fechar o círculo
        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            color=colors[i],
            label=row['symbol'],
            alpha=0.8,
        )
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    # Configurar o radar
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title('Comparação Multidimensional de Dispersão', fontsize=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig('figures/comparacao_radar_dispersao.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Boxplot para comparar a distribuição dos preços
    plt.figure(figsize=(16, 10))
    symbols = df_dispersao['symbol'].tolist()
    boxplot_data = []
    labels = []

    for symbol in symbols:
        prices = dados_originais[dados_originais['symbol'] == symbol]['close'].dropna()
        if not prices.empty:
            boxplot_data.append(prices)
            labels.append(symbol)

    plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
    plt.title('Distribuição de Preços por Criptomoeda', fontsize=16)
    plt.xlabel('Criptomoeda', fontsize=12)
    plt.ylabel('Preço de Fechamento', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/boxplot_comparativo_precos.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Visualizações comparativas de dispersão salvas com sucesso.")


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
    calculate_summary_statistics(filtered_data, "summary_statistics.csv", symbols)
    calculate_dispersion_measures(filtered_data, "dispersion_statistics.csv", symbols)

    # Análise detalhada de dispersão e volatilidade entre as criptomoedas
    df_dispersao = analisar_medidas_dispersao(
        filtered_data, "analise_dispersao.csv", symbols
    )
    visualizar_comparacao_dispersao(df_dispersao, filtered_data)
    logger.info("Análise de dispersão e volatilidade concluída com sucesso.")

    logger.info("Estatísticas calculadas e salvas com sucesso.")

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
