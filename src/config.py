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
