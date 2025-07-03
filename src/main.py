"""
Este script orquestra o pipeline completo de análise de criptomoedas.

Ele executa as seguintes etapas:
1. Carrega os dados brutos e os processa.
2. Calcula as features necessárias para análise.
3. Treina os modelos de previsão.
4. Simula estratégias de investimento com base nos modelos treinados.

O objetivo é fornecer uma análise detalhada e simulações para auxiliar na tomada de decisões.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_load import main as data_load_main
from src.features import main as features_main
from src.models import main as models_main
from src.simulate import main as simulate_main
from src.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    try:
        logging.info("Executando o pipeline de análise de criptomoedas...")

        # Carregar dados
        data = data_load_main()

        # Calcular features
        data_with_features = features_main()

        # Treinar modelos
        models, data_calculate = models_main()

        # Simular investimentos
        simulate_main(data_calculate, models=models)

        logging.info("Pipeline concluído com sucesso.")
    except Exception as e:
        logging.critical(f"Erro crítico durante a execução: {e}")

