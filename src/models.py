import time
import functools
import pandas as pd
import os
import datetime
import sys
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import USE_TIMING, INITIAL_CAPITAL
from src.utils import timing
from src.utils import filter_symbols, sanitize_symbol, get_current_datetime, calculate_correlation_coefficients

@timing
def walk_forward_prediction(model, X, y, min_train_size=30):
    """
    Gera previsões cronológicas usando a abordagem Walk-Forward (janela expansível),
    com logging de progresso.
    """
    # Inicializa a lista para armazenar as previsões
    predictions = []

    # Configura o TimeSeriesSplit para o loop de validação
    n_splits = len(X) - min_train_size
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)
    
    print(f"Iniciando predição Walk-Forward para {model.__class__.__name__}...")

    # Inicializa o saldo acumulado fora do loop
    accumulated_balance = INITIAL_CAPITAL

    # Loop explícito para permitir futura adição de logging ou outras operações
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # Clona o modelo para garantir que não haja vazamento de estado
        model_clone = clone(model)
        
        # Separa os dados de treino e teste para a iteração atual
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]
        
        # Treina o modelo com os dados históricos até o momento
        model_clone.fit(X_train, y_train)
        
        # Faz a previsão para o próximo dia e a armazena
        pred = model_clone.predict(X_test)[0]
        predictions.append(pred)
        
        # Verifica se a previsão indica compra (previsão maior que o preço atual)
        current_price = y.iloc[test_index[0]]
        action = "Compra realizada" if pred > current_price else "Sem compra"
        
        # Calcula o saldo acumulado (simulação simples)
        if i == 0:
            accumulated_balance = INITIAL_CAPITAL
        else:
            previous_price = y.iloc[test_index[0] - 1]
            accumulated_balance *= (current_price / previous_price) if action == "Compra realizada" else 1
        
        # Imprime o status do progresso a cada 1 dia ou no último dia
        #if (i + 1) % 1 == 0 or (i + 1) == n_splits:
        # (Removido: cálculo de porcentagem de acerto/erro dentro do loop)

    # Calcula a porcentagem de erro e acerto após o loop
    correct_predictions = 0
    total_predictions = 0
    total_buys = 0
    y_true = y.iloc[min_train_size:].values
    preds = np.array(predictions)
    correct_predictions = np.sum((y_true[:-1] < y_true[1:]) == (preds[:-1] < preds[1:]))
    total_predictions = len(preds) - 1
    if total_predictions > 0:
        accuracy_percentage = (correct_predictions / total_predictions) * 100
        error_percentage = 100 - accuracy_percentage
    else:
        accuracy_percentage = 0
        error_percentage = 100
    total_days = len(predictions)
    y_future = y.iloc[min_train_size:min_train_size + total_days].values
    total_buys = np.sum(np.array(predictions) > y_future)

    print(f"\nPorcentagem de acerto: {accuracy_percentage:.2f}%")
    print(f"Porcentagem de erro: {error_percentage:.2f}%")
    print(f"Total de compras sugeridas: {total_buys}/{total_days} dias")

    # Retorna o array de previsões
    return np.array(predictions)


@timing
def k_fold_validation(model, X, y, n_splits=5):
    """
    Aplica validação K-Fold ao modelo e retorna MSE médio.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        model_clone = clone(model)
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)