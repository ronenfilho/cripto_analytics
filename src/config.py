from pathlib import Path
import os
from dotenv import load_dotenv  # type: ignore

# Diretório raiz
ROOT_DIR = Path(__file__).resolve().parent.parent

# Diretórios de dados
DATASET_DIR = ROOT_DIR / "data"
RAW_DATA = DATASET_DIR / "raw"
PROCESSED_DATA = DATASET_DIR / "processed"
PROCESSED_FILE = PROCESSED_DATA / "cripto_data.csv"

# Variáveis de configuração
# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Define a quantidade de dias da simulação com base no .env
TEST_PERIOD_DAYS = int(os.getenv("TEST_PERIOD_DAYS", "30"))
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")

# Variáveis para análise estatística
EXPECTED_RETURN = float(os.getenv("EXPECTED_RETURN", "1.5"))  # em porcentagem
SIGNIFICANCE_LEVEL = float(os.getenv("SIGNIFICANCE_LEVEL", "0.05"))

# Atualiza os modelos ativos com base no .env
MODELS = {
    "LinearRegression": os.getenv("USE_LINEAR_REGRESSION", "True").lower() == "true",
    "MLPRegressor": os.getenv("USE_MLP_REGRESSOR", "True").lower() == "true",
    "PolynomialRegression": os.getenv("USE_POLYNOMIAL_REGRESSION", "True").lower()
    == "true",
    # "DecisionTreeRegressor": os.getenv("USE_DECISION_TREE_REGRESSOR", "True").lower() == "true",
    # "RandomForestRegressor": os.getenv("USE_RANDOM_FOREST_REGRESSOR", "True").lower() == "true",
    # "XGBRegressor": os.getenv("USE_XGB_REGRESSOR", "True").lower() == "true",
    # "CatBoostRegressor": os.getenv("USE_CATBOOST_REGRESSOR", "True").lower() == "true",
    # "LightGBMRegressor": os.getenv("USE_LIGHTGBM_REGRESSOR", "True").lower() == "true",
    # "AdaBoostRegressor": os.getenv("USE_ADA_BOOST_REGRESSOR", "True").lower() == "true",
    # "GradientBoostingRegressor": os.getenv("USE_GRADIENT_BOOSTING_REGRESSOR", "True").lower() == "true",
}

# Define se o modelo PolynomialRegression será utilizado com base no .env
# USE_POLYNOMIAL_REGRESSION = str(os.getenv("USE_POLYNOMIAL_REGRESSION", "True")).lower() == "true"

# Atualiza para suportar intervalo de graus para PolynomialRegression
POLYNOMIAL_DEGREE_RANGE = os.getenv("POLYNOMIAL_DEGREE_RANGE", "2,5").split(",")

# Define os símbolos para simulação com base no .env
SYMBOLS_TO_SIMULATE = os.getenv("SYMBOLS_TO_SIMULATE", "BTC/USDT").split(",")

# Define o capital inicial para simulação com base no .env
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))

# Define se o timing deve ser utilizado com base no .env
USE_TIMING = os.getenv("USE_TIMING", "True").lower() == "true"

# Define o nível de logging com base no .env
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Configurações de visualização de gráficos
FIGURE_DPI = int(os.getenv("FIGURE_DPI", "150"))
FIGURE_FORMAT = os.getenv("FIGURE_FORMAT", "png").lower()
FIGURE_SIZE_WIDTH = float(os.getenv("FIGURE_SIZE_WIDTH", "12"))
FIGURE_SIZE_HEIGHT = float(os.getenv("FIGURE_SIZE_HEIGHT", "8"))

# Configurações avançadas de análise
CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", "5"))
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "rmse").lower()
ANOVA_GROUPING_CRITERIA = os.getenv(
    "ANOVA_GROUPING_CRITERIA", "volatility,mean_return"
).split(",")
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", "0.5"))
