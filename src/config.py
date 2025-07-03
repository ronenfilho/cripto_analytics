from pathlib import Path
import os
from dotenv import load_dotenv # type: ignore
from pathlib import Path

# Diretório raiz
ROOT_DIR = Path(__file__).resolve().parent.parent

# Diretórios de dados
DATASET_DIR = ROOT_DIR / 'data'
RAW_DATA = DATASET_DIR / 'raw'
PROCESSED_DATA = DATASET_DIR / 'processed'
PROCESSED_FILE = PROCESSED_DATA / 'cripto_data.csv'

# Variáveis de configuração
# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Define a quantidade de dias da simulação com base no .env
TEST_PERIOD_DAYS = int(os.getenv("TEST_PERIOD_DAYS", "30"))
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(',')

# Atualiza os modelos ativos com base no .env
MODELS = {
    "LinearRegression": os.getenv("USE_LINEAR_REGRESSION", "True").lower() == "true",
    "MLPRegressor": os.getenv("USE_MLP_REGRESSOR", "True").lower() == "true",
    "PolynomialRegression": os.getenv("USE_POLYNOMIAL_REGRESSION", "True").lower() == "true",
    #"DecisionTreeRegressor": os.getenv("USE_DECISION_TREE_REGRESSOR", "True").lower() == "true",
    #"RandomForestRegressor": os.getenv("USE_RANDOM_FOREST_REGRESSOR", "True").lower() == "true",
    #"XGBRegressor": os.getenv("USE_XGB_REGRESSOR", "True").lower() == "true",
    #"CatBoostRegressor": os.getenv("USE_CATBOOST_REGRESSOR", "True").lower() == "true",
    #"LightGBMRegressor": os.getenv("USE_LIGHTGBM_REGRESSOR", "True").lower() == "true",
    #"AdaBoostRegressor": os.getenv("USE_ADA_BOOST_REGRESSOR", "True").lower() == "true",
    #"GradientBoostingRegressor": os.getenv("USE_GRADIENT_BOOSTING_REGRESSOR", "True").lower() == "true",        
}

# Define se o modelo PolynomialRegression será utilizado com base no .env
#USE_POLYNOMIAL_REGRESSION = str(os.getenv("USE_POLYNOMIAL_REGRESSION", "True")).lower() == "true"

# Atualiza para suportar intervalo de graus para PolynomialRegression
POLYNOMIAL_DEGREE_RANGE = os.getenv("POLYNOMIAL_DEGREE_RANGE", "2,5").split(',')

# Define o símbolo específico para simulação com base no .env
SYMBOL_TO_SIMULATE = os.getenv("SYMBOL_TO_SIMULATE", "BTC/USDT")

# Define o capital inicial para simulação com base no .env
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))

# Define se o timing deve ser utilizado com base no .env
USE_TIMING = os.getenv("USE_TIMING", "True").lower() == "true"

# Define o nível de logging com base no .env
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()


