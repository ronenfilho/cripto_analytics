import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import warnings
import logging

# Ignora todos os warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_FILE, SYMBOLS
from src.utils import timing, filter_symbols, setup_logging

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)

# Ignora logs do Matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

    grouped_stats = data.groupby('symbol')['close'].describe()
    logger.debug(f"Medidas resumo calculadas: {grouped_stats}")

    logger.info("Função calculate_summary_statistics concluída com sucesso.")
    return grouped_stats

@timing
def generate_visualizations(data: pd.DataFrame):
    """
    Gera boxplots e histogramas dos preços de fechamento agrupados por 'symbol'.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Iniciando a função generate_visualizations.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    symbols = data['symbol'].unique()
    logger.info(f"Símbolos encontrados: {symbols}")

    os.makedirs('figures', exist_ok=True)

    for symbol in symbols:
        logger.info(f"Processando o símbolo: {symbol}")
        symbol_data = data[data['symbol'] == symbol]

        logger.debug(f"Tamanho dos dados para {symbol}: {symbol_data.shape}")

        # Boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(symbol_data['close'])
        plt.title(f"Boxplot - {symbol}")
        plt.xlabel("Symbol")
        plt.ylabel("Preço de Fechamento")

        sanitized_symbol = symbol.replace('/', '_')
        plt.savefig(f"figures/{sanitized_symbol}_boxplot.png", dpi=150)
        plt.close()
        logger.info(f"Boxplot salvo para o símbolo: {symbol}")

        # Histograma
        plt.figure(figsize=(10, 6))
        plt.hist(symbol_data['close'], bins=20, alpha=0.7)
        plt.title(f"Histograma - {symbol}")
        plt.xlabel("Preço de Fechamento")
        plt.ylabel("Frequência")
        plt.savefig(f"figures/{sanitized_symbol}_histogram.png", dpi=150)
        plt.close()
        logger.info(f"Histograma salvo para o símbolo: {symbol}")

        logger.debug(f"Histograma salvo para o símbolo: {symbol}")

@timing
def compare_variability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza uma análise comparativa da variabilidade entre as moedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com a variabilidade (desvio padrão) de cada moeda.
    """

    
    logger.info('#################################################################')
    logger.info(f"Analisar a variabilidade entre as criptomoedas com base nas medidas de dispersão.")
    logger.info('#################################################################')
    

    logger.info("Iniciando a função compare_variability.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    variability = data.groupby('symbol')['close'].std().reset_index()
    variability.columns = ['symbol', 'variability']
    logger.debug(f"Variabilidade calculada: {variability}")

    logger.info("Função compare_variability concluída com sucesso.")
    return variability

@timing
def plot_variability(data: pd.DataFrame):
    """
    Cria e salva um gráfico de barras com a análise da variabilidade entre as criptomoedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Iniciando a função plot_variability.")
    logger.debug(f"Tamanho do DataFrame recebido: {data.shape}")
    logger.debug(f"Colunas do DataFrame: {data.columns.tolist()}")

    variability = compare_variability(data)

    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.bar(variability['symbol'], variability['variability'], color='skyblue')
    plt.yscale('log')
    plt.title("Variabilidade entre as criptomoedas (Escala Logarítmica)")
    plt.xlabel("Symbol")
    plt.ylabel("Desvio Padrão (Variabilidade)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/variability_analysis_log.png", dpi=150)
    plt.close()
    logger.info("Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'")

    # Gráfico de barras com variabilidade normalizada
    normalized_variability = variability.copy()
    normalized_variability['variability'] /= normalized_variability['variability'].max()

    plt.figure(figsize=(12, 8))
    plt.bar(normalized_variability['symbol'], normalized_variability['variability'])
    plt.title("Análise de Variabilidade Normalizada")
    plt.xlabel("Símbolo")
    plt.ylabel("Variabilidade Normalizada")
    plt.savefig("figures/variability_analysis_normalized.png", dpi=150)
    plt.close()
    logger.info("Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'")

    logger.info("Função plot_variability concluída com sucesso.")

@timing
def plot_normalized_variability(data: pd.DataFrame):
    """
    Cria e salva um gráfico de barras com a variabilidade normalizada e anotações.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    logger.info("Criando gráfico de variabilidade normalizada entre as criptomoedas.")

    variability = compare_variability(data)
    max_variability = variability['variability'].max()
    variability['normalized_variability'] = variability['variability'] / max_variability * 100

    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(variability['symbol'], variability['normalized_variability'], color='skyblue')
    plt.title("Variabilidade Normalizada entre as Criptomoedas")
    plt.xlabel("Symbol")
    plt.ylabel("Variabilidade Normalizada (%)")
    plt.xticks(rotation=45)

    for bar, value in zip(bars, variability['normalized_variability']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("figures/variability_analysis_normalized.png", dpi=150)
    plt.close()
    logger.info("Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'.")

@timing
def plot_price_trends_by_month_year(data: pd.DataFrame):
    """
    Cria e salva gráficos de linha por símbolo agrupados por mês e ano com o preço de fechamento destacando média, mediana e moda.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """

    
    logger.info('#################################################################')
    logger.info(f"Construir gráfico de linha com o preço de fechamento destacando a média, mediana e moda ao longo do tempo.")
    logger.info('#################################################################')
    

    # Garante que a pasta 'figures' existe
    os.makedirs('figures', exist_ok=True)
    data['date'] = pd.to_datetime(data['date'])
    data['month_year'] = data['date'].dt.to_period('M')

    symbols = data['symbol'].unique()

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        grouped_data = symbol_data.groupby('month_year')['close'].agg(['mean', 'median', lambda x: x.mode()[0] if not x.mode().empty else None, 'last']).reset_index()
        grouped_data.columns = ['month_year', 'mean', 'median', 'mode', 'close']
        sanitized_symbol = symbol.replace('/', '_')

        plt.figure(figsize=(16, 10))
        plt.plot(grouped_data['month_year'].astype(str), grouped_data['mean'], label='Média', color='green', linestyle='--')
        plt.plot(grouped_data['month_year'].astype(str), grouped_data['median'], label='Mediana', color='orange', linestyle='--')
        plt.plot(grouped_data['month_year'].astype(str), grouped_data['mode'], label='Moda', color='red', linestyle='--')
        plt.plot(grouped_data['month_year'].astype(str), grouped_data['close'], label='Preço de Fechamento', color='blue')

        plt.title(f"Tendências de Preço de Fechamento por Mês/Ano - {symbol}")
        plt.xlabel("Mês/Ano")
        plt.ylabel("Preço de Fechamento")
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"figures/{sanitized_symbol}_price_trends_month_year.png", dpi=150)
        plt.close()
        logger.debug(f"Gráfico de tendências de preço salvo para o símbolo {symbol}.")

@timing
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features baseadas em séries temporais para previsão de preços.

    Args:
        data (pd.DataFrame): DataFrame com colunas ['date', 'symbol', 'close'].

    Returns:
        pd.DataFrame: DataFrame com novas colunas de features.
    """
    logger.info("Calculando features baseadas em séries temporais para previsão de preços.")

    data = data.copy()
    data['return_1d'] = data['close'].pct_change(1)
    data['symbol_original'] = data['symbol']

    data = data.groupby('symbol', group_keys=False).apply(
        lambda df: df.assign(
            mean_7d=df['close'].rolling(window=7).mean(),
            std_7d=df['close'].rolling(window=7).std(),
            return_7d=df['close'].pct_change(7),
            rolling_max_7d=df['close'].rolling(window=7).max(),
            rolling_min_7d=df['close'].rolling(window=7).min(),
            momentum_7d=df['close'] - df['close'].shift(7),
            volatility_7d=df['return_1d'].rolling(window=7).std()
        ),
        include_groups=False
    )

    data = data.rename(columns={'symbol_original': 'symbol'})

    expected_cols = ['mean_7d', 'std_7d']
    for col in expected_cols:
        if col not in data.columns:
            logger.error(f"A coluna '{col}' não foi calculada corretamente.")
            raise ValueError(f"A coluna '{col}' não foi calculada corretamente.")

    logger.debug("Features calculadas com sucesso.")
    return data

def main():
    """Função principal para ser chamada por outros métodos.
    Carrega os dados processados, filtra conforme os símbolos selecionados, calcula estatísticas resumo e gera visualizações.
    """
    
    data = pd.read_csv(PROCESSED_FILE)

    symbols = SYMBOLS    

    logger.info(f"Símbolo: {symbols if symbols else 'Todos'}")
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
    plot_variability(filtered_data)
    logger.info("Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'")
    logger.info("Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'")

    # Gera e salva o gráfico de variabilidade normalizada
    plot_normalized_variability(filtered_data)
    logger.info("Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'")
    logger.info("Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'")

    # Gera e salva gráficos de tendências de preço por mês/ano
    plot_price_trends_by_month_year(filtered_data)
    logger.info("Gráficos de tendências de preço por mês/ano salvos na pasta 'figures'")
    logger.info("Gráficos de tendências de preço por mês/ano salvos na pasta 'figures'")

if __name__ == "__main__":
    main()
    main()
