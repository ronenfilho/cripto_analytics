import pandas as pd
import os
from scipy.stats import ttest_1samp
from pathlib import Path
import csv
from typing import List, Dict
import logging
import sys


# Importa o caminho do arquivo de configuração
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA, EXPECTED_RETURN, SIGNIFICANCE_LEVEL
from src.utils import setup_logging

# Configura o logger
setup_logging()
logger = logging.getLogger(__name__)

def load_simulation_results() -> pd.DataFrame:
    """Carrega os dados de retorno do arquivo simulation_results_days.csv

    Returns:
        pd.DataFrame: DataFrame com os dados de simulação
    """
    file_path = PROCESSED_DATA / "simulation_results_days.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_csv(file_path)

def hypothesis_test_by_strategy(data: pd.DataFrame, expected_return: float, significance_level: float = 0.05) -> Dict[str, Dict[str, any]]:
    """Realiza um teste de hipótese para verificar se o retorno médio esperado é superior ou igual a um valor definido pelo usuário,
    considerando cada combinação de símbolo e estratégia.

    Args:
        data (pd.DataFrame): DataFrame contendo as colunas ['symbol', 'strategy', 'return_pct'].
        expected_return (float): Retorno esperado médio (em porcentagem).
        significance_level (float): Nível de significância para o teste (default: 0.05).

    Returns:
        Dict[str, Dict[str, any]]: Resultado do teste para cada combinação de símbolo e estratégia, incluindo o p-valor e a conclusão.
    """
    results = {}

    for (symbol, strategy), group in data.groupby(["symbol", "strategy"]):
        returns = group["return_pct"].dropna()

        # Realiza o teste t de uma amostra (teste unilateral à esquerda)
        # H0: μ ≥ expected_return (retorno médio é maior ou igual ao esperado)
        # H1: μ < expected_return (retorno médio é menor que o esperado)
        t_stat, p_value_bilateral = ttest_1samp(returns, expected_return / 100)
        
        # Para teste unilateral à esquerda (μ < x), usamos:
        # Se t_stat < 0: p_value = p_value_bilateral / 2
        # Se t_stat ≥ 0: p_value = 1 - (p_value_bilateral / 2)
        if t_stat < 0:
            p_value = p_value_bilateral / 2
        else:
            p_value = 1 - (p_value_bilateral / 2)

        if t_stat < 0 and p_value < significance_level:
            conclusion = "Rejeitar H0 - O retorno médio é significativamente menor que o esperado"
        else:
            conclusion = "Não rejeitar H0 - O retorno médio é maior ou igual ao esperado"

        results[(symbol, strategy)] = {
            "t_stat": t_stat,
            "p_value": p_value,
            "conclusion": conclusion,
        }

    return results

def save_hypothesis_test_results(results: List[Dict[str, any]], filename: str = "analysis_hypothesis_test_results.csv") -> None:
    """Salva os resultados do teste de hipótese em um arquivo .csv na pasta processed.

    Args:
        results (List[Dict[str, any]]): Lista de dicionários contendo os resultados do teste de hipótese.
        filename (str): Nome do arquivo .csv a ser salvo.

    Returns:
        None
    """
    filepath = PROCESSED_DATA / filename
    headers = ["Cryptocurrency", "Strategy", "Expected Mean (%)", "Achieved Mean (%)", "t-statistic", "p-value", "Conclusion"]

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "Cryptocurrency": result["symbol"],
                "Strategy": result["strategy"],
                "Expected Mean (%)": f"{result['expected_return']:.4f}",
                "Achieved Mean (%)": f"{result['achieved_mean']:.4f}",
                "t-statistic": f"{result['t_stat']:.4f}",
                "p-value": f"{result['p_value']:.4f}",
                "Conclusion": result["conclusion"]
            })

    logging.info(f"Resultados do teste de hipótese salvos em: {filepath}")

def main() -> None:
    # Carrega os dados de simulação
    data = load_simulation_results()
    
    # Filtra apenas os dados onde o investimento foi realizado
    data_filtered = data[data['investment_made'] == 'Yes']
    
    expected_return = EXPECTED_RETURN
    significance_level = SIGNIFICANCE_LEVEL

    logging.info("Análise Estatística - Teste de Hipótese")
    logging.info(f"Retorno Esperado: {expected_return:.2f}%")
    logging.info(f"Nível de Significância: {significance_level:.2f}")
    logging.info(f"Total de registros: {len(data)}")
    logging.info(f"Registros com investimento realizado: {len(data_filtered)}")
    logging.info("=" * 70)
    
    logging.info("DESCRIÇÃO DO TESTE DE HIPÓTESE:")
    logging.info("H0 (Hipótese Nula): O retorno médio diário é maior ou igual ao valor esperado (μ ≥ x%)")
    logging.info("H1 (Hipótese Alternativa): O retorno médio diário é menor que o valor esperado (μ < x%)")
    logging.info(f"Teste: t-test unilateral à esquerda (μ < {expected_return:.2f}%)")
    logging.info("Interpretação:")
    logging.info(f"  - p-valor < {significance_level:.2f}: Rejeitar H0 (retorno é significativamente menor que o esperado)")
    logging.info(f"  - p-valor ≥ {significance_level:.2f}: Não rejeitar H0 (retorno é maior ou igual ao esperado)")
    logging.info("  - Estatística t < 0: Retorno observado < esperado")
    logging.info("  - Estatística t ≥ 0: Retorno observado ≥ esperado")
    logging.info("=" * 70)

    # Teste por símbolo e estratégia
    results_by_strategy = hypothesis_test_by_strategy(data_filtered, expected_return, significance_level)
    
    logging.info("Resultados por símbolo e estratégia:")
    for (symbol, strategy), result in results_by_strategy.items():
        # Calcula a média alcançada para este grupo
        group_data = data_filtered[(data_filtered['symbol'] == symbol) & (data_filtered['strategy'] == strategy)]
        achieved_mean = group_data['return_pct'].mean() * 100  # Converte para porcentagem

        logging.info(f"Criptomoeda: {symbol}, Estratégia: {strategy}")
        logging.info(f"  Média Esperada: {expected_return:.4f}%")
        logging.info(f"  Média Alcançada: {achieved_mean:.4f}%")
        logging.info(f"  Estatística t: {result['t_stat']:.4f}")
        logging.info(f"  p-valor: {result['p_value']:.4f}")
        logging.info(f"  Conclusão: {result['conclusion']}")
        logging.info("-" * 40)

    # Salva os resultados em um arquivo .csv
    results_to_save = [
        {
            "symbol": symbol,
            "strategy": strategy,
            "expected_return": expected_return,
            "achieved_mean": data_filtered[(data_filtered['symbol'] == symbol) & (data_filtered['strategy'] == strategy)]['return_pct'].mean() * 100,
            "t_stat": result["t_stat"],
            "p_value": result["p_value"],
            "conclusion": result["conclusion"],
        }
        for (symbol, strategy), result in results_by_strategy.items()
    ]
    save_hypothesis_test_results(results_to_save)

if __name__ == "__main__":
    main()