import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_FILE
from src.utils import timing, filter_symbols

'''
Análise de Performance (com K-Fold): Manteremos sua validação K-Fold original para gerar a tabela de RMSE e provar a performance geral dos modelos.

Simulação de Investimento (com Walk-Forward): Para gerar as previsões necessárias para a simulação, usaremos uma técnica chamada Walk-Forward Validation (ou Validação de Janela Expansível). É o método correto para testar estratégias em séries temporais, pois funciona assim:

Treina com dados do dia 1 ao 100, prevê o dia 101.
Treina com dados do dia 1 ao 101, prevê o dia 102.
E assim por diante...
Isso garante que cada previsão seja feita usando apenas dados do passado, simulando perfeitamente um ambiente real.
'''

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


def k_fold_validation(model, X, y, n_splits=5):
    """
    Aplica validação K-Fold ao modelo e retorna MSE médio.
    
    Args:
        model: Estimador scikit-learn.
        X (pd.DataFrame): Features.
        y (pd.Series): Alvo.
        n_splits (int): Número de folds.

    Returns:
        float: Erro médio quadrático médio.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)


if __name__ == "__main__":
    data = pd.read_csv(PROCESSED_FILE)

    symbols = []    
    #symbols = ['BTC/USDT', 'ETH/USDT']
    #symbols = ['BTC/USDT']

    print(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Filtra os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula features
    data_calculate = calculate_features(filtered_data)
    #print(f"Features calculadas:\n{data_calculate[['date', 'symbol', 'close', 'mean_7d', 'std_7d']].head()}")

    # Remove linhas com valores ausentes
    features_to_check = ['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']
    data_calculate = data_calculate.dropna(subset=features_to_check)
    print(f"Dados após remoção de NaNs:\n{data_calculate[features_to_check].head()}")

    # Define X e y
    X = data_calculate[['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']]
    y = data_calculate['close']

    models = {
        "LinearRegression": LinearRegression(),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

    results = []

    for name, model in models.items():
        mse = k_fold_validation(model, X, y)
        rmse = np.sqrt(mse)
        results.append({"Modelo": name, "MSE": mse, "RMSE": rmse})

    print('\n')
    print('#################################################################')
    print("Resultados da Validação K-Fold (Legenda):")
    print('#################################################################')    
    print(f" - Erro médio quadrático (MSE)")
    print(f" - Raiz do erro médio quadrático (RMSE)")    
    print('\n')

    results_df = pd.DataFrame(results)
    print("Comparação de Modelos:")
    print(results_df)
