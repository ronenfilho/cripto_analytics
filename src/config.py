from pathlib import Path
import os
from dotenv import load_dotenv # type: ignore
from pathlib import Path


# Carrega o .env apenas uma vez ao importar
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

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
}

# Atualiza para suportar intervalo de graus para PolynomialRegression
POLYNOMIAL_DEGREE_RANGE = os.getenv("POLYNOMIAL_DEGREE_RANGE", "2,5").split(',')

USE_POLYNOMIAL_REGRESSION = str(os.getenv("USE_POLYNOMIAL_REGRESSION", "True")).lower() == "true"

SYMBOL_TO_SIMULATE = os.getenv("SYMBOL_TO_SIMULATE", "BTC/USDT")

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))
