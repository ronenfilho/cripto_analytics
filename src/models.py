import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_FILE
from src.utils import timing, filter_symbols


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Função para calcular features baseadas na série temporal.
    Calcula features como média móvel, desvio padrão e correlação entre moedas.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados das criptomoedas.

    Returns:
        pd.DataFrame: DataFrame com as features calculadas.
    """
    # Calcula features em uma única atribuição para melhor performance
    data = data.assign(
        mean_7d=data['close'].rolling(window=7).mean(),
        std_7d=data['close'].rolling(window=7).std()
    )

    # Certifica-se de que as colunas estão presentes antes de acessar
    if 'mean_7d' not in data.columns or 'std_7d' not in data.columns:
        raise ValueError("As colunas 'mean_7d' e 'std_7d' não foram calculadas corretamente.")

    return data

def k_fold_validation(model, X, y, n_splits=5):
    """
    Aplica validação K-Fold ao modelo.

    Args:
        model: Modelo a ser validado.
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variável alvo.
        n_splits (int): Número de divisões para o K-Fold.

    Returns:
        float: Erro médio quadrático (MSE) médio.
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

    #symbols = []    
    #symbols = ['BTC/USDT', 'ETH/USDT']
    symbols = ['BTC/USDT']

    print(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Use filter_symbols para filtrar os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula features
    data_calculate = calculate_features(filtered_data)
    print(f"Dados com features calculadas:\n{data_calculate.head()}") 

    # Ignora linhas incompletas ao calcular as features
    data_calculate = data_calculate.dropna(subset=['mean_7d', 'std_7d'])
    print(f"Dados após remoção de linhas incompletas:\n{data_calculate.head()}")

    # Certifica-se de que o modelo está sendo treinado apenas com dados completos
    X = data_calculate[['mean_7d', 'std_7d']]
    y = data_calculate['close']
    model = LinearRegression()

    mse = k_fold_validation(model, X, y)
    print(f"Erro médio quadrático (MSE) médio: {mse}")