import pandas as pd
import os
import sys
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import logging
from joblib import dump, load
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    INITIAL_CAPITAL,
    SYMBOLS,
    MODELS,
    POLYNOMIAL_DEGREE_RANGE,
    PROCESSED_FILE,
    PROCESSED_DATA,
)
from src.features import calculate_features
from src.utils import timing, filter_symbols, setup_logging

setup_logging()

# Configura o logger
logger = logging.getLogger(__name__)


@timing
def walk_forward_prediction(
    model: object, X: pd.DataFrame, y: pd.Series, min_train_size: int = 30
) -> np.ndarray:
    """
    Gera previsões cronológicas usando a abordagem Walk-Forward (janela expansível),
    com logging de progresso.
    """
    # Inicializa a lista para armazenar as previsões
    predictions = []
    
    # Armazena as previsões originais para debug e consistência
    original_predictions = []

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
        original_predictions.append(pred)  # Salva a previsão original

        # Verifica se a previsão indica compra (previsão maior que o preço atual)
        current_price = y.iloc[test_index[0]]
        action = "Compra realizada" if pred > current_price else "Sem compra"

        # Calcula o saldo acumulado (simulação simples)
        if i < len(y) - min_train_size - 1:  # Verifica se há um próximo preço
            next_price = y.iloc[test_index[0] + 1]
            if action == "Compra realizada":
                # Aplica o retorno baseado no próximo preço
                return_rate = (next_price - current_price) / current_price
                accumulated_balance *= (1 + return_rate)
            # Se não comprou, o saldo permanece igual

        # Imprime o status do progresso a cada 1 dia ou no último dia
        if (i + 1) % 1 == 0 or (i + 1) == n_splits:
            logger.info(
                f"Dia {i + 1}/{n_splits}: Previsão = {pred:.2f}, Preço atual = {current_price:.2f}, Ação = {action}, Saldo acumulado = {accumulated_balance:.2f}"
            )

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
    y_future = y.iloc[min_train_size : min_train_size + total_days].values
    total_buys = np.sum(np.array(predictions) > y_future)

    logger.info(f"Porcentagem de acerto: {accuracy_percentage:.2f}%")
    logger.info(f"Porcentagem de erro: {error_percentage:.2f}%")
    logger.info(f"Total de compras sugeridas: {total_buys}/{total_days} dias")

    # Retorna o array de previsões
    return np.array(predictions)


@timing
def k_fold_validation(
    model: object, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> float:
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
        logger.error(
            f"Erro: Arquivo não encontrado em '{PROCESSED_FILE}'. Ajuste a variável no script."
        )
        sys.exit(1)

    # --- PARTE 1: Análise de Performance com K-Fold ---

    logger.info("#################################################################")
    logger.info("PARTE 1: Análise de Performance com K-Fold (Legenda):")
    logger.info("#################################################################")
    logger.info(" - Erro médio quadrático (MSE)")
    logger.info(" - Raiz do erro médio quadrático (RMSE)")

    logger.info(f"Símbolo: {symbols if symbols else 'Todos'}")

    # Filtra os dados conforme os símbolos selecionados
    filtered_data = filter_symbols(data, symbols if symbols else None)

    # Calcula features
    data_calculate = calculate_features(filtered_data)

    # Remove linhas com valores ausentes
    features_to_check = [
        "mean_7d",
        "std_7d",
        "return_7d",
        "momentum_7d",
        "volatility_7d",
    ]
    data_calculate = data_calculate.dropna(subset=features_to_check)
    logger.debug(
        f"Dados após remoção de NaNs:\n{data_calculate[features_to_check].head()}"
    )

    # Define X e y
    X = data_calculate[
        ["mean_7d", "std_7d", "return_7d", "momentum_7d", "volatility_7d"]
    ]
    y = data_calculate["close"]

    # Filtra os modelos ativos
    active_models = {name: model for name, model in MODELS.items() if model}

    # Cria instâncias dos modelos ativos
    models = {}
    for name in active_models.keys():
        if name == "LinearRegression":
            models[name] = LinearRegression()
        elif name == "MLPRegressor":
            models[name] = MLPRegressor(
                hidden_layer_sizes=(50,), max_iter=400, random_state=42
            )
        elif name == "PolynomialRegression":
            poly_degree_range = range(
                int(POLYNOMIAL_DEGREE_RANGE[0]), int(POLYNOMIAL_DEGREE_RANGE[1]) + 1
            )
            for degree in poly_degree_range:
                models[f"PolynomialRegression_degree_{degree}"] = make_pipeline(
                    PolynomialFeatures(degree), LinearRegression()
                )
        elif name == "ElasticNet":
            models[name] = ElasticNet(random_state=42)
        elif name == "Ridge":
            models[name] = Ridge(random_state=42)
        elif name == "Lasso":
            models[name] = Lasso(random_state=42)
        elif name == "RandomForestRegressor":
            models[name] = RandomForestRegressor(random_state=42)
        elif name == "GradientBoostingRegressor":
            models[name] = GradientBoostingRegressor(random_state=42)
        elif name == "SVR":
            models[name] = SVR()

    results = []

    for name, model in models.items():
        mse = k_fold_validation(model, X, y)
        rmse = np.sqrt(mse)
        results.append({"Modelo": name, "MSE": mse, "RMSE": rmse})

    results_df = pd.DataFrame(results)
    logger.info("Comparação de Modelos:")
    logger.info(results_df)

    return models, data_calculate


@timing
def create_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features avançadas para melhorar a capacidade preditiva dos modelos.
    
    Args:
        data (pd.DataFrame): DataFrame com dados de preço e features básicas
        
    Returns:
        pd.DataFrame: DataFrame com features avançadas adicionadas
    """
    logger.info("Criando features avançadas para melhorar a previsão...")
    
    # Clone dos dados para evitar modificar o original
    df = data.copy()
    
    # Organiza por símbolo e data
    df = df.sort_values(['symbol', 'date'])
    
    # Agrupa por símbolo para calcular features específicas por moeda
    result = []
    
    for symbol, group in df.groupby('symbol'):
        # Ordena por data
        group = group.sort_values('date')
        
        # Features técnicas avançadas
        # RSI (Relative Strength Index) - 14 dias
        delta = group['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # Evita divisão por zero
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        group['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema12 - ema26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
        group['macd_hist'] = group['macd'] - group['macd_signal']
        
        # Bollinger Bands
        group['bb_middle'] = group['close'].rolling(window=20).mean()
        bb_std = group['close'].rolling(window=20).std()
        group['bb_upper'] = group['bb_middle'] + (bb_std * 2)
        group['bb_lower'] = group['bb_middle'] - (bb_std * 2)
        group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / group['bb_middle']
        group['bb_pct'] = (group['close'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'])
        
        # Features de volume
        group['volume_change'] = group['Volume USDT'].pct_change()
        group['volume_ma10'] = group['Volume USDT'].rolling(window=10).mean()
        group['volume_ma10_ratio'] = group['Volume USDT'] / group['volume_ma10']
        
        # Volatilidade em diferentes janelas
        for window in [3, 5, 10, 20]:
            group[f'volatility_{window}d'] = group['close'].pct_change().rolling(window=window).std()
        
        # Features de momentum em diferentes janelas
        for window in [3, 5, 14, 21]:
            group[f'momentum_{window}d'] = group['close'] - group['close'].shift(window)
            group[f'return_{window}d'] = group['close'].pct_change(window)
            group[f'mean_{window}d'] = group['close'].rolling(window=window).mean()
            
        # ROC (Rate of Change)
        for window in [5, 10, 20]:
            group[f'roc_{window}d'] = (group['close'] / group['close'].shift(window) - 1) * 100
        
        # Features de preço relativas a médias móveis
        group['price_to_ma50'] = group['close'] / group['close'].rolling(window=50).mean()
        group['price_to_ma200'] = group['close'] / group['close'].rolling(window=200).mean()
        
        # Cruzamentos de médias móveis (sinais de compra/venda)
        ma_fast = group['close'].rolling(window=10).mean()
        ma_slow = group['close'].rolling(window=30).mean()
        group['ma_crossover'] = np.where(ma_fast > ma_slow, 1, -1)
        
        # Adiciona features cíclicas de tempo (dia da semana, mês, etc.)
        dates = pd.to_datetime(group['date'])
        group['day_of_week'] = dates.dt.dayofweek
        group['day_of_week_sin'] = np.sin(2 * np.pi * group['day_of_week'] / 7)
        group['day_of_week_cos'] = np.cos(2 * np.pi * group['day_of_week'] / 7)
        group['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
        group['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
        
        # Adiciona o grupo ao resultado
        result.append(group)
    
    # Combina todos os grupos com as novas features
    df_result = pd.concat(result)
    
    # Remove linhas com valores NaN (necessário devido aos cálculos de rolling window)
    df_result = df_result.dropna()
    
    logger.info(f"Features avançadas criadas com sucesso. Total de features: {len(df_result.columns)}")
    return df_result


def main():
    """
    Função principal para ser chamada por outros métodos.
    """
    models, data_calculate = run_training_data()
    return models, data_calculate


if __name__ == "__main__":
    main()
