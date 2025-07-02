import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_FILE, SYMBOLS, MODELS, POLYNOMIAL_DEGREE_RANGE, SYMBOL_TO_SIMULATE, INITIAL_CAPITAL, TEST_PERIOD_DAYS
from src.utils import timing, filter_symbols, sanitize_symbol, get_current_datetime, calculate_correlation_coefficients, calculate_standard_error_between_mlp_and_best, determine_best_equation, calculate_standard_error
from src.models import k_fold_validation, walk_forward_prediction

'''
Análise de Performance (com K-Fold): Manteremos sua validação K-Fold original para gerar a tabela de RMSE e provar a performance geral dos modelos.

Simulação de Investimento (com Walk-Forward): Para gerar as previsões necessárias para a simulação, usaremos uma técnica chamada Walk-Forward Validation (ou Validação de Janela Expansível). É o método correto para testar estratégias em séries temporais, pois funciona assim:

Treina com dados do dia 1 ao 100, prevê o dia 101.
Treina com dados do dia 1 ao 101, prevê o dia 102.
E assim por diante...
Isso garante que cada previsão seja feita usando apenas dados do passado, simulando perfeitamente um ambiente real.
'''

@timing
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features baseadas em séries temporais para previsão de preços.
    
    Features incluídas:
    - Média móvel (7 dias)
    - Desvio padrão (7 dias)
    - Retornos (1 e 7 dias)
    - Máximo e mínimo (7 dias)
    - Momentum (diferença entre preços)
    - Volatilidade (desvio dos retornos)
    
    Args:
        data (pd.DataFrame): DataFrame com colunas ['date', 'symbol', 'close'].
    
    Returns:
        pd.DataFrame: DataFrame com novas colunas de features.
    """
    data = data.copy()

    # Calcula o retorno diário
    data['return_1d'] = data['close'].pct_change(1)

    # Agrupa por símbolo (para suportar múltiplas moedas)
    data['symbol_original'] = data['symbol']  # salva o valor antes do apply

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

    # Renomeia a coluna de volta para 'symbol'
    data = data.rename(columns={'symbol_original': 'symbol'})

    # Verificação básica de colunas essenciais
    expected_cols = ['mean_7d', 'std_7d']
    for col in expected_cols:
        if col not in data.columns:
            raise ValueError(f"A coluna '{col}' não foi calculada corretamente.")

    return data

@timing
def simulate_returns(y_test: pd.Series, y_pred: np.ndarray, initial_capital: float = 1000.0) -> list:
    """Simula os retornos de uma estratégia de investimento."""

    capital = initial_capital
    capital_evolution = [initial_capital]
    y_test_values = y_test.values

    # Ajusta o cálculo de capital_evolution para incluir o capital inicial
    daily_returns = np.where(y_pred[1:] > y_test_values[:-1], y_test_values[1:] / y_test_values[:-1], 1)
    capital_evolution = np.concatenate(([initial_capital], initial_capital * np.cumprod(daily_returns)))

    return capital_evolution.tolist()

@timing
def run_investment_simulation(data: pd.DataFrame, symbol_to_simulate: str, models: dict, initial_capital: float = 1000.0, test_period_days: int = 365):
    """
    Executa o fluxo completo de simulação de investimento para um ativo e plota o resultado.
    Simula os últimos 'test_period_days' dias.
    """
    print("\n" + "#"*60)
    print(f"PARTE 2: SIMULAÇÃO DE INVESTIMENTO PARA {symbol_to_simulate}")
    print("#"*60)

    single_symbol_data = filter_symbols(data, [symbol_to_simulate])
    
    # Garante que os dados estão ordenados por data em ordem crescente
    single_symbol_data['date'] = pd.to_datetime(single_symbol_data['date'])
    single_symbol_data = single_symbol_data.sort_values('date', ascending=True)
    
    data_with_features = calculate_features(single_symbol_data)
    data_processed = data_with_features.dropna()

    features_cols = ['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']
    X_sim = data_processed[features_cols]
    y_sim = data_processed['close']
    dates_sim = pd.to_datetime(data_processed['date'])
    
    min_train_size = len(X_sim) - test_period_days
    if min_train_size < 100:
        print("Erro: Não há dados suficientes para uma simulação com o período de teste solicitado.")
        return

    y_test_sim = y_sim.iloc[min_train_size:]
    dates_test_sim = dates_sim.iloc[min_train_size:]

    # Ordena por data para garantir ordem cronológica correta
    sorted_idx = np.argsort(dates_test_sim.values)
    dates_test_sim = dates_test_sim.iloc[sorted_idx]
    y_test_sim = y_test_sim.iloc[sorted_idx]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for name, model in models.items():
        y_pred_walk_forward = walk_forward_prediction(model, X_sim, y_sim, min_train_size)
        y_pred_walk_forward = np.array(y_pred_walk_forward)[sorted_idx]
        capital_evolution = simulate_returns(y_test_sim, y_pred_walk_forward, initial_capital)
        ax.plot(dates_test_sim, capital_evolution, label=f'Estratégia {name}')
        print(f"Capital final com {name}: U${capital_evolution[-1]:.2f}")

    # Estratégia "Buy and Hold"
    y_test_values = y_test_sim.values
    hold_evolution = [initial_capital * (price / y_test_values[0]) for price in y_test_values]
    ax.plot(dates_test_sim, hold_evolution, label='Estratégia Buy and Hold', linestyle='--')
    print(f"Capital final com Buy and Hold: U${hold_evolution[-1]:.2f}")

    # Configurações do Gráfico
    ax.set_title(f'Evolução do Capital - Simulação Walk-Forward ({symbol_to_simulate})', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Capital Acumulado (U$)', fontsize=12)
    ax.legend(fontsize=10)
    formatter = mticker.FormatStrFormatter('U$%.0f')
    ax.yaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # Garante que a pasta 'figures' existe
    os.makedirs('figures', exist_ok=True)

    file_name = f"{get_current_datetime()}_{sanitize_symbol(symbol_to_simulate)}_investment_simulation_{test_period_days}_days.png"

    # Salva o gráfico na pasta figures antes de mostrar
    plt.savefig(f"figures/{file_name}", dpi=150, bbox_inches='tight')
    
    #plt.show()
    plt.close()

    # Ajusta o tamanho de 'dates_test_sim' para coincidir com 'capital_evolution'
    dates_test_sim = dates_test_sim[:len(capital_evolution)]

    # Ajusta 'capital_evolution' para garantir que tenha a quantidade informada de elementos
    if len(capital_evolution) < len(dates_test_sim):
        missing_elements = len(dates_test_sim) - len(capital_evolution)
        capital_evolution = np.append(capital_evolution, [capital_evolution[-1]] * missing_elements)

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

    print("\n")
    print("#################################################################")
    print("Diferença entre Buy and Hold e Modelos (MLP, etc.):")
    print("#################################################################")
    print("\n")
    print("Buy and Hold:")
    print(" - Compra no início e mantém até o final.")
    print(" - Evolução proporcional ao preço inicial e final.")
    print("\n")
    print("Modelos (MLP, etc.):")
    print(" - Usa previsões para ajustar posições diariamente.")
    print(" - Evolução depende da precisão das previsões.")
    print("\n")

    # Adiciona explicação sobre o cálculo do lucro com o modelo
    print("\n")
    print("#################################################################")
    print("Computar o lucro obtido com seu modelo:")
    print("#################################################################")
    print("\n")
    print("Caso tenha investido U$ 1,000.00 no primeiro dia de operação:")
    print(" - Refazendo investimentos de todo o saldo acumulado diariamente.")
    print(" - Apenas se a previsão do valor de fechamento do próximo dia for superior ao do dia atual.")
    print("\n")

@timing
def run_training_data():
    """
    Executa o fluxo completo de treinamento e validação dos modelos.
    """
    
    symbols = SYMBOLS

    try:
        data = pd.read_csv(PROCESSED_FILE)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{PROCESSED_FILE}'. Ajuste a variável no script.")
        sys.exit(1)

    # --- PARTE 1: Análise de Performance com K-Fold ---
    print('\n')
    print('#################################################################')
    print("PARTE 1: Análise de Performance com K-Fold (Legenda):")
    print('#################################################################')    
    print(f" - Erro médio quadrático (MSE)")
    print(f" - Raiz do erro médio quadrático (RMSE)")    
    print('\n')     

    print(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Filtra os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula features
    data_calculate = calculate_features(filtered_data)

    # Remove linhas com valores ausentes
    features_to_check = ['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']
    data_calculate = data_calculate.dropna(subset=features_to_check)
    print(f"Dados após remoção de NaNs:\n{data_calculate[features_to_check].head()}")

    # Define X e y
    X = data_calculate[['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']]
    y = data_calculate['close']

    # Filtra os modelos ativos
    active_models = {name: model for name, model in MODELS.items() if model}

    # Cria instâncias dos modelos ativos
    models = {}
    for name in active_models.keys():
        if name == "LinearRegression":
            models[name] = LinearRegression()
        elif name == "MLPRegressor":
            models[name] = MLPRegressor(hidden_layer_sizes=(50,), max_iter=400, random_state=42)
        elif name == "PolynomialRegression":
            poly_degree_range = range(int(POLYNOMIAL_DEGREE_RANGE[0]), int(POLYNOMIAL_DEGREE_RANGE[1]) + 1)
            for degree in poly_degree_range:
                models[f"PolynomialRegression_degree_{degree}"] = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    results = []

    for name, model in models.items():
        mse = k_fold_validation(model, X, y)
        rmse = np.sqrt(mse)
        results.append({"Modelo": name, "MSE": mse, "RMSE": rmse})    

    results_df = pd.DataFrame(results)
    print("Comparação de Modelos:")
    print(results_df)

    return models, data_calculate

# Adiciona métodos para análise dos modelos

def plot_scatter_diagram(models, X, y, save_path='figures/scatter_diagram.png'):
    """
    Gera um diagrama de dispersão para todos os modelos.
    Salva o gráfico na pasta figures.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        y_pred = model_clone.predict(X)
        ax.scatter(y, y_pred, label=name, alpha=0.6)

    ax.set_title('Diagrama de Dispersão - Modelos', fontsize=16)
    ax.set_xlabel('Valores Reais', fontsize=12)
    ax.set_ylabel('Valores Preditos', fontsize=12)
    ax.legend(fontsize=10)

    # Atualiza o nome do arquivo para o scatter diagram
    scatter_file_name = f"{get_current_datetime()}_scatter_diagram.png"
 
    # Salva o gráfico na pasta figures antes de mostrar
    plt.savefig(f"figures/{scatter_file_name}", dpi=150, bbox_inches='tight')
    
    #plt.show()
    plt.close()

if __name__ == "__main__":
    # --- PARTE 1: Treinamento e Validação dos Modelos ---
    models, data = run_training_data()

    # Define X e y para análise
    X = data[['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']]
    y = data['close']

    # --- PARTE 2: Chamada para a Simulação de Investimento ---
    print("\n")
    run_investment_simulation(data=data, symbol_to_simulate=SYMBOL_TO_SIMULATE, models=models, initial_capital=INITIAL_CAPITAL, test_period_days=TEST_PERIOD_DAYS)

    # --- PARTE 3: Análise dos Modelos ---
    print("\n")
    print("#################################################################")
    print("Análise dos Modelos:")
    print("#################################################################")
    print("\n")

    # Gera o diagrama de dispersão
    plot_scatter_diagram(models, X, y)

    # Calcula os coeficientes de correlação
    correlations = calculate_correlation_coefficients(models, X, y)
    print("Coeficientes de Correlação:")
    for name, corr in correlations.items():
        print(f" - {name}: {corr:.4f}")

    # Determina a melhor equação
    best_model_name, best_score = determine_best_equation(models, X, y)
    print(f"Melhor Modelo: {best_model_name} com score {best_score:.4f}")

    # Calcula o erro padrão
    errors = calculate_standard_error(models, X, y)
    print("Erro Padrão:")
    for name, error in errors.items():
        print(f" - {name}: {error:.4f}")

    # Calcula o erro padrão entre MLP e o melhor regressor
    error_between_mlp_and_best = calculate_standard_error_between_mlp_and_best(models, X, y)
    print(f"Erro Padrão entre MLP e {best_model_name}: {error_between_mlp_and_best:.4f}")
