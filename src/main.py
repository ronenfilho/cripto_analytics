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
import argparse
from dotenv import set_key

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging

setup_logging()


def update_env_variable(key: str, value: str):
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    set_key(dotenv_path, key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de previsão de preços de criptomoedas"
    )

    parser.add_argument(
        "--data",
        action="store_true",
        help="(1) Executa a carga e transformação do dados",
    )
    parser.add_argument(
        "--features", action="store_true", help="(2) Executa o cálculo de features"
    )
    parser.add_argument(
        "--model", action="store_true", help="(3) Executa o treinamento de modelos"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="(4) Executa a simulação de investimento",
    )
    # TODO
    parser.add_argument("--days", type=int, help="Quantidade de dias para a simulação")
    parser.add_argument("--capital", type=float, help="Capital inicial para simulação")
    parser.add_argument(
        "--crypto",
        type=str,
        help="Símbolos das criptomoedas a serem analisadas. Exemplo: BTC/USDT,ETH/USDT. Todos os símbolos disponíveis na pasta raw serão utilizados se não for especificado. Permitidos: BCH/USDT,BTC/USDT,DASH/USDT,EOS/USDT,ETC/USDT,ETH/USDT,LTC/USDT,XMR/USDT,XRP/USDT,ZRX/USDT",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Número de folds para validação cruzada (K-Fold)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        logging.info("Executando o pipeline de análise de criptomoedas...")

        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
        # Atualizar variáveis no .env dinamicamente se fornecidas
        if args.crypto:
            update_env_variable("SYMBOLS", str(args.crypto))
        if args.days:
            update_env_variable("TEST_PERIOD_DAYS", str(args.days))
        if args.capital:
            update_env_variable("INITIAL_CAPITAL", str(args.capital))

        # Reimportar config após ajustes
        import importlib
        import config

        importlib.reload(config)

        from src.data_load import main as data_load_main
        from src.features import main as features_main
        from src.models import main as models_main
        from src.simulate import main as simulate_main

        # Verificar quais etapas executar com base nos argumentos
        if args.crypto or args.model or args.days or args.capital:
            logging.info(
                "Atualizando variáveis de ambiente com base nos argumentos fornecidos."
            )

        if args.data:
            logging.info("Executando etapa de carregamento de dados...")
            data_load_main()

        if args.features:
            logging.info("Executando etapa de cálculo de features...")
            data_load_main()
            features_main()

        if args.model:
            logging.info("Executando etapa de treinamento de modelos...")
            data_load_main()
            features_main()
            models, data_calculate = models_main()

        if args.simulate:
            logging.info("Executando etapa de simulação de investimentos...")
            data_load_main()
            features_main()
            models, data_calculate = models_main()
            simulate_main(data_calculate, models=models)

        # Verifica se argumentos foram passados, caso contrário, processa todos
        if not any([args.data, args.features, args.model, args.simulate]):
            logging.info(
                "Nenhum argumento fornecido do pipeline. Processando todas as etapas do pipeline..."
            )
            data_load_main()
            features_main()
            models, data_calculate = models_main()
            simulate_main(data_calculate, models=models)

        logging.info("Pipeline concluído com sucesso.")

    except Exception as e:
        logging.critical(f"Erro crítico durante a execução: {e}")
