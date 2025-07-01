import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_FILE
from src.utils import timing

@timing
def calculate_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula medidas resumo e de dispersão para os preços de fechamento agrupados por 'symbol'.
    Se um símbolo for informado, retorna apenas para esse símbolo.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
        symbol (str, opcional): Símbolo da criptomoeda. Se None, calcula para todos.

    Returns:
        pd.DataFrame: DataFrame com as medidas resumo e de dispersão agrupadas por 'symbol' ou para o símbolo informado.
    """
    print('#################################################################')
    print(f"Cálculo de medidas resumo e de dispersão para as criptomoedas.")  
    print('#################################################################')  

    grouped_stats = data.groupby('symbol')['close'].describe()
    return grouped_stats

@timing
def generate_visualizations(data: pd.DataFrame):
    """
    Gera boxplots e histogramas dos preços de fechamento agrupados por 'symbol'.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """

    print('#################################################################')
    print(f"Geração de boxplots e histogramas dos preços de fechamento.")
    print('#################################################################')
    
    symbols = data['symbol'].unique()

    # Garante que a pasta 'figures' existe
    os.makedirs('figures', exist_ok=True)

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]

        # Boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(symbol_data['close'])
        plt.title(f"Boxplot - {symbol}")
        plt.xlabel("Symbol")
        plt.ylabel("Preço de Fechamento")
        
        # Substitui '/' por '_' no nome do símbolo
        sanitized_symbol = symbol.replace('/', '_')

        # Salva os gráficos com o nome sanitizado
        plt.savefig(f"figures/{sanitized_symbol}_boxplot.png", dpi=150)
        plt.close()

        # Histograma
        plt.figure(figsize=(10, 6))
        plt.hist(symbol_data['close'], bins=20, alpha=0.7)
        plt.title(f"Histograma - {symbol}")
        plt.xlabel("Preço de Fechamento")
        plt.ylabel("Frequência")
        plt.savefig(f"figures/{sanitized_symbol}_histogram.png", dpi=150)
        plt.close()

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

@timing
def compare_variability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza uma análise comparativa da variabilidade entre as moedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com a variabilidade (desvio padrão) de cada moeda.
    """

    print('#################################################################')
    print(f"Analisar a variabilidade entre as criptomoedas com base nas medidas de dispersão.")
    print('#################################################################')

    variability = data.groupby('symbol')['close'].std().reset_index()
    variability.columns = ['symbol', 'variability']
    return variability

def plot_variability(data: pd.DataFrame):
    """
    Cria e salva um gráfico de barras com a análise da variabilidade entre as criptomoedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    variability = compare_variability(data)

    # Garante que a pasta 'figures' existe
    os.makedirs('figures', exist_ok=True)

    # Gráfico de barras com escala logarítmica
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

def plot_normalized_variability(data: pd.DataFrame):
    """
    Cria e salva um gráfico de barras com a variabilidade normalizada e anotações.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.
    """
    variability = compare_variability(data)

    # Normaliza os valores de variabilidade
    max_variability = variability['variability'].max()
    variability['normalized_variability'] = variability['variability'] / max_variability * 100

    # Garante que a pasta 'figures' existe
    os.makedirs('figures', exist_ok=True)

    # Gráfico de barras com variabilidade normalizada
    plt.figure(figsize=(12, 8))
    bars = plt.bar(variability['symbol'], variability['normalized_variability'], color='skyblue')
    plt.title("Variabilidade Normalizada entre as Criptomoedas")
    plt.xlabel("Symbol")
    plt.ylabel("Variabilidade Normalizada (%)")
    plt.xticks(rotation=45)

    # Adiciona anotações nas barras
    for bar, value in zip(bars, variability['normalized_variability']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("figures/variability_analysis_normalized.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    data = pd.read_csv(PROCESSED_FILE)

    symbols = []
    #symbols = ['BTC/USDT']
    #symbols = ['BTC/USDT', 'ETH/USDT']    

    print(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Use filter_symbols para filtrar os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula estatísticas resumo e de dispersão para as criptomoedas filtradas
    grouped_stats = calculate_summary_statistics(filtered_data)
    print(grouped_stats)

    # Gera boxplots e histogramas dos preços de fechamento para os símbolos filtrados
    generate_visualizations(filtered_data)

    # Análise comparativa da variabilidade entre as moedas
    variability = compare_variability(filtered_data)
    print(variability)

    # Gera e salva o gráfico de variabilidade
    plot_variability(filtered_data)
    print("Gráfico de variabilidade salvo em 'figures/variability_analysis_log.png'")

    # Gera e salva o gráfico de variabilidade normalizada
    plot_normalized_variability(filtered_data)
    print("Gráfico de variabilidade normalizada salvo em 'figures/variability_analysis_normalized.png'")
