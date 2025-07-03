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
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL, SYMBOLS, MODELS, POLYNOMIAL_DEGREE_RANGE, PROCESSED_FILE
from src.features import calculate_features
from src.utils import timing, filter_symbols, setup_logging

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)

@timing
def walk_forward_prediction(model: object, X: pd.DataFrame, y: pd.Series, min_train_size: int = 30) -> np.ndarray:
    """
    Gera previsões cronológicas usando a abordagem Walk-Forward (janela expansível),
    com logging de progresso.
    """
    # Inicializa a lista para armazenar as previsões
    predictions = []

    # Configura o TimeSeriesSplit para o loop de validação
    n_splits = len(X) - min_train_size
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)
    
    logger.info(f"Iniciando predição Walk-Forward para {model.__class__.__name__}...")

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
        if (i + 1) % 1 == 0 or (i + 1) == n_splits:
            logger.info(f"Dia {i + 1}/{n_splits}: Previsão = {pred:.2f}, Preço atual = {current_price:.2f}, Ação = {action}, Saldo acumulado = {accumulated_balance:.2f}")

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

    logger.info(f"\nPorcentagem de acerto: {accuracy_percentage:.2f}%")
    logger.info(f"Porcentagem de erro: {error_percentage:.2f}%")
    logger.info(f"Total de compras sugeridas: {total_buys}/{total_days} dias")

    # Retorna o array de previsões
    return np.array(predictions)


@timing
def k_fold_validation(model: object, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
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


@timing
def run_training_data() -> tuple[dict, pd.DataFrame]:
    """
    Executa o fluxo completo de treinamento e validação dos modelos.
    """
    
    symbols = SYMBOLS

    try:
        data = pd.read_csv(PROCESSED_FILE)
    except FileNotFoundError:
        logger.error(f"Erro: Arquivo não encontrado em '{PROCESSED_FILE}'. Ajuste a variável no script.")
        sys.exit(1)

    # --- PARTE 1: Análise de Performance com K-Fold ---
    
    logger.info('#################################################################')
    logger.info("PARTE 1: Análise de Performance com K-Fold (Legenda):")
    logger.info('#################################################################')    
    logger.info(f" - Erro médio quadrático (MSE)")
    logger.info(f" - Raiz do erro médio quadrático (RMSE)")    
    

    logger.info(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Filtra os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula features
    data_calculate = calculate_features(filtered_data)

    # Remove linhas com valores ausentes
    features_to_check = ['mean_7d', 'std_7d', 'return_7d', 'momentum_7d', 'volatility_7d']
    data_calculate = data_calculate.dropna(subset=features_to_check)
    logger.debug(f"Dados após remoção de NaNs:\n{data_calculate[features_to_check].head()}")

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
    logger.info("Comparação de Modelos:")
    logger.info(results_df)

    return models, data_calculate

def main():
    """
    Função principal para ser chamada por outros métodos.
    """
    models, data_calculate = run_training_data()
    return models, data_calculate


if __name__ == "__main__":
    main()