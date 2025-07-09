import pandas as pd
import os
from scipy.stats import ttest_1samp, f_oneway, ttest_ind
from pathlib import Path
import csv
from typing import List, Dict
import logging
import sys
import numpy as np
from itertools import combinations


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

def anova_analysis_cryptocurrencies(data: pd.DataFrame, significance_level: float = 0.05) -> Dict[str, any]:
    """
    Realiza análise ANOVA para comparar retornos médios diários entre criptomoedas.
    
    Args:
        data (pd.DataFrame): DataFrame contendo colunas ['symbol', 'return_pct'].
        significance_level (float): Nível de significância para o teste (default: 0.05).
    
    Returns:
        Dict[str, any]: Resultados da análise ANOVA e teste post hoc.
    """
    # Filtra apenas dados com investimento realizado
    data_filtered = data[data['investment_made'] == 'Yes'].copy()
    
    # Agrupa retornos por criptomoeda
    groups = []
    symbols = []
    for symbol, group in data_filtered.groupby('symbol'):
        returns = group['return_pct'].dropna()
        if len(returns) > 1:  # Precisa de pelo menos 2 observações
            groups.append(returns.values)
            symbols.append(symbol)
    
    if len(groups) < 2:
        return {"error": "Necessário pelo menos 2 grupos com dados suficientes para ANOVA"}
    
    # Realiza ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    # Interpretação do resultado
    is_significant = p_value < significance_level
    anova_conclusion = "Há diferença significativa entre as criptomoedas" if is_significant else "Não há diferença significativa entre as criptomoedas"
    
    results = {
        "anova_f_statistic": f_stat,
        "anova_p_value": p_value,
        "is_significant": is_significant,
        "anova_conclusion": anova_conclusion,
        "symbols_analyzed": symbols,
        "post_hoc": None
    }
    
    # Se significativo, realiza teste post hoc simples
    if is_significant:
        # Prepara dados para teste post hoc
        groups_data = {}
        
        for symbol, group in data_filtered.groupby('symbol'):
            returns = group['return_pct'].dropna()
            if len(returns) > 1:
                groups_data[symbol] = returns.values
        
        # Realiza teste post hoc simples
        post_hoc_results = simple_pairwise_ttest(groups_data, significance_level)
        
        results["post_hoc"] = {
            "method": "Pairwise t-test",
            "pairwise_comparisons": post_hoc_results
        }
    
    return results

def anova_analysis_grouped_cryptocurrencies(data: pd.DataFrame, grouping_criterion: str = "volatility", 
                                          significance_level: float = 0.05) -> Dict[str, any]:
    """
    Realiza análise ANOVA para comparar retornos médios diários entre grupos de criptomoedas.
    
    Args:
        data (pd.DataFrame): DataFrame contendo dados das criptomoedas.
        grouping_criterion (str): Critério de agrupamento ('volatility', 'volume', 'mean_return').
        significance_level (float): Nível de significância para o teste (default: 0.05).
    
    Returns:
        Dict[str, any]: Resultados da análise ANOVA e teste post hoc para grupos.
    """
    # Filtra apenas dados com investimento realizado
    data_filtered = data[data['investment_made'] == 'Yes'].copy()
    
    # Calcula métricas por criptomoeda para agrupamento
    crypto_metrics = {}
    for symbol, group in data_filtered.groupby('symbol'):
        returns = group['return_pct'].dropna()
        if len(returns) > 1:
            crypto_metrics[symbol] = {
                "mean_return": returns.mean(),
                "volatility": returns.std(),
                "volume": len(returns)  # Proxy para volume (número de observações)
            }
    
    if len(crypto_metrics) < 2:
        return {"error": "Necessário pelo menos 2 criptomoedas com dados suficientes"}
    
    # Define grupos baseado no critério escolhido
    if grouping_criterion == "volatility":
        values = [metrics["volatility"] for metrics in crypto_metrics.values()]
        median_value = np.median(values)
        groups_dict = {}
        for symbol, metrics in crypto_metrics.items():
            group_name = "High Volatility" if metrics["volatility"] > median_value else "Low Volatility"
            if group_name not in groups_dict:
                groups_dict[group_name] = []
            groups_dict[group_name].append(symbol)
    
    elif grouping_criterion == "mean_return":
        values = [metrics["mean_return"] for metrics in crypto_metrics.values()]
        median_value = np.median(values)
        groups_dict = {}
        for symbol, metrics in crypto_metrics.items():
            group_name = "High Return" if metrics["mean_return"] > median_value else "Low Return"
            if group_name not in groups_dict:
                groups_dict[group_name] = []
            groups_dict[group_name].append(symbol)
    
    else:  # volume
        values = [metrics["volume"] for metrics in crypto_metrics.values()]
        median_value = np.median(values)
        groups_dict = {}
        for symbol, metrics in crypto_metrics.items():
            group_name = "High Volume" if metrics["volume"] > median_value else "Low Volume"
            if group_name not in groups_dict:
                groups_dict[group_name] = []
            groups_dict[group_name].append(symbol)
    
    # Coleta retornos por grupo
    group_returns = []
    group_names = []
    
    for group_name, symbols in groups_dict.items():
        group_data = data_filtered[data_filtered['symbol'].isin(symbols)]
        returns = group_data['return_pct'].dropna()
        if len(returns) > 1:
            group_returns.append(returns.values)
            group_names.append(group_name)
    
    if len(group_returns) < 2:
        return {"error": "Necessário pelo menos 2 grupos com dados suficientes para ANOVA"}
    
    # Realiza ANOVA
    f_stat, p_value = f_oneway(*group_returns)
    
    # Interpretação do resultado
    is_significant = p_value < significance_level
    anova_conclusion = f"Há diferença significativa entre os grupos ({grouping_criterion})" if is_significant else f"Não há diferença significativa entre os grupos ({grouping_criterion})"
    
    results = {
        "grouping_criterion": grouping_criterion,
        "groups": groups_dict,
        "anova_f_statistic": f_stat,
        "anova_p_value": p_value,
        "is_significant": is_significant,
        "anova_conclusion": anova_conclusion,
        "post_hoc": None
    }
    
    # Se significativo, realiza teste post hoc simples
    if is_significant:
        # Prepara dados para teste post hoc
        groups_data = {}
        
        for group_name, symbols in groups_dict.items():
            group_data = data_filtered[data_filtered['symbol'].isin(symbols)]
            returns = group_data['return_pct'].dropna()
            if len(returns) > 1:
                groups_data[group_name] = returns.values
        
        # Realiza teste post hoc simples
        post_hoc_results = simple_pairwise_ttest(groups_data, significance_level)
        
        results["post_hoc"] = {
            "method": "Pairwise t-test",
            "pairwise_comparisons": post_hoc_results
        }
    
    return results

# Alternativa simples para teste post hoc sem statsmodels
def simple_pairwise_ttest(groups_data: Dict[str, np.ndarray], significance_level: float = 0.05) -> List[Dict[str, any]]:
    """Realiza teste t pareado simples entre grupos."""
    results = []
    group_names = list(groups_data.keys())
    
    for group1, group2 in combinations(group_names, 2):
        t_stat, p_value = ttest_ind(groups_data[group1], groups_data[group2])
        
        results.append({
            "group1": group1,
            "group2": group2,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < significance_level,
            "mean_diff": np.mean(groups_data[group1]) - np.mean(groups_data[group2])
        })
    
    return results

def save_anova_individual_cryptocurrencies(anova_results: Dict[str, any], filename: str = "analysis_anova_individual_cryptos.txt") -> None:
    """Salva os resultados da ANOVA para criptomoedas individuais em um arquivo .txt.

    Args:
        anova_results (Dict[str, any]): Resultados da ANOVA para criptomoedas individuais.
        filename (str): Nome do arquivo .txt a ser salvo.

    Returns:
        None
    """
    filepath = PROCESSED_DATA / filename
    
    with open(filepath, mode="w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write("ANÁLISE ANOVA - CRIPTOMOEDAS INDIVIDUAIS\n")
        file.write("=" * 80 + "\n\n")
        
        file.write(f"Tipo de Análise: ANOVA\n")
        file.write(f"Descrição do Teste: Comparação de retornos médios diários entre criptomoedas individuais\n\n")
        
        file.write("CRIPTOMOEDAS ANALISADAS:\n")
        file.write("-" * 40 + "\n")
        for i, symbol in enumerate(anova_results['symbols_analyzed'], 1):
            file.write(f"{i}. {symbol}\n")
        file.write("\n")
        
        file.write("RESULTADOS DA ANOVA:\n")
        file.write("-" * 40 + "\n")
        file.write(f"Estatística F: {anova_results['anova_f_statistic']:.4f}\n")
        file.write(f"Valor p: {anova_results['anova_p_value']:.4f}\n")
        file.write(f"Significativo: {'Sim' if anova_results['is_significant'] else 'Não'}\n")
        file.write(f"Conclusão: {anova_results['anova_conclusion']}\n\n")
        
        file.write("INTERPRETAÇÃO:\n")
        file.write("-" * 40 + "\n")
        if anova_results['is_significant']:
            file.write("H0 (Hipótese Nula): Os retornos médios são iguais entre todas as criptomoedas\n")
            file.write("H1 (Hipótese Alternativa): Pelo menos uma criptomoeda tem retorno médio diferente\n")
            file.write("Resultado: REJEITAR H0 - Há diferenças significativas entre as criptomoedas\n\n")
            
            if anova_results['post_hoc']:
                file.write("ANÁLISE POST HOC (Teste t pareado):\n")
                file.write("-" * 40 + "\n")
                file.write("Para identificar quais criptomoedas diferem entre si:\n\n")
                for comparison in anova_results['post_hoc']['pairwise_comparisons']:
                    file.write(f"{comparison['group1']} vs {comparison['group2']}:\n")
                    file.write(f"  Diferença de Média: {comparison['mean_diff']:.4f}\n")
                    file.write(f"  Estatística t: {comparison['t_statistic']:.4f}\n")
                    file.write(f"  Valor p: {comparison['p_value']:.4f}\n")
                    file.write(f"  Significativo: {'Sim' if comparison['significant'] else 'Não'}\n\n")
        else:
            file.write("H0 (Hipótese Nula): Os retornos médios são iguais entre todas as criptomoedas\n")
            file.write("H1 (Hipótese Alternativa): Pelo menos uma criptomoeda tem retorno médio diferente\n")
            file.write("Resultado: NÃO REJEITAR H0 - Não há diferenças significativas entre as criptomoedas\n")
            file.write("Análise Post Hoc: Não aplicável (ANOVA não significativa)\n\n")
        
        file.write("=" * 80 + "\n")
        file.write("Análise concluída com sucesso\n")
        file.write("=" * 80 + "\n")
    
    logging.info(f"ANOVA individual cryptocurrencies results saved to: {filepath}")

def save_anova_post_hoc_individual(anova_results: Dict[str, any], filename: str = "analysis_anova_post_hoc_individual.csv") -> None:
    """Salva os resultados do teste post hoc para criptomoedas individuais em um arquivo .csv.

    Args:
        anova_results (Dict[str, any]): Resultados da ANOVA para criptomoedas individuais.
        filename (str): Nome do arquivo .csv a ser salvo.

    Returns:
        None
    """
    if not anova_results['is_significant'] or not anova_results['post_hoc']:
        logging.info("No significant results for post hoc test - individual cryptocurrencies")
        return
    
    filepath = PROCESSED_DATA / filename
    
    # Dados do teste post hoc
    post_hoc_data = []
    for comparison in anova_results['post_hoc']['pairwise_comparisons']:
        post_hoc_data.append({
            "Cryptocurrency_1": comparison['group1'],
            "Cryptocurrency_2": comparison['group2'],
            "Mean_Difference": f"{comparison['mean_diff']:.4f}",
            "T_Statistic": f"{comparison['t_statistic']:.4f}",
            "P_Value": f"{comparison['p_value']:.4f}",
            "Significant": "Yes" if comparison['significant'] else "No",
            "Test_Method": "Pairwise t-test"
        })
    
    # Salva resultados post hoc
    headers = ["Cryptocurrency_1", "Cryptocurrency_2", "Mean_Difference", "T_Statistic", "P_Value", "Significant", "Test_Method"]
    
    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(post_hoc_data)
    
    logging.info(f"Post hoc individual cryptocurrencies results saved to: {filepath}")

def save_anova_grouped_cryptocurrencies(anova_results: Dict[str, any], filename: str = "analysis_anova_grouped_cryptos.txt") -> None:
    """Salva os resultados da ANOVA para grupos de criptomoedas em um arquivo .txt.

    Args:
        anova_results (Dict[str, any]): Resultados da ANOVA para grupos de criptomoedas.
        filename (str): Nome do arquivo .txt a ser salvo.

    Returns:
        None
    """
    filepath = PROCESSED_DATA / filename
    
    with open(filepath, mode="w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write("ANÁLISE ANOVA - GRUPOS DE CRIPTOMOEDAS\n")
        file.write("=" * 80 + "\n\n")
        
        file.write(f"Tipo de Análise: ANOVA\n")
        file.write(f"Descrição do Teste: Comparação de retornos médios diários entre grupos de criptomoedas\n")
        file.write(f"Critério de Agrupamento: {anova_results['grouping_criterion']}\n\n")
        
        file.write("GRUPOS FORMADOS:\n")
        file.write("-" * 40 + "\n")
        for group_name, symbols in anova_results['groups'].items():
            file.write(f"{group_name}: {', '.join(symbols)}\n")
        file.write("\n")
        
        file.write("RESULTADOS DA ANOVA:\n")
        file.write("-" * 40 + "\n")
        file.write(f"Estatística F: {anova_results['anova_f_statistic']:.4f}\n")
        file.write(f"Valor p: {anova_results['anova_p_value']:.4f}\n")
        file.write(f"Significativo: {'Sim' if anova_results['is_significant'] else 'Não'}\n")
        file.write(f"Conclusão: {anova_results['anova_conclusion']}\n\n")
        
        if anova_results['is_significant'] and anova_results['post_hoc']:
            file.write("ANÁLISE POST HOC (Teste t pareado):\n")
            file.write("-" * 40 + "\n")
            for comparison in anova_results['post_hoc']['pairwise_comparisons']:
                file.write(f"{comparison['group1']} vs {comparison['group2']}:\n")
                file.write(f"  Diferença de Média: {comparison['mean_diff']:.4f}\n")
                file.write(f"  Estatística t: {comparison['t_statistic']:.4f}\n")
                file.write(f"  Valor p: {comparison['p_value']:.4f}\n")
                file.write(f"  Significativo: {'Sim' if comparison['significant'] else 'Não'}\n\n")
        else:
            file.write("ANÁLISE POST HOC: Não aplicável (ANOVA não significativa)\n\n")
        
        file.write("=" * 80 + "\n")
        file.write("Análise concluída com sucesso\n")
        file.write("=" * 80 + "\n")
    
    logging.info(f"ANOVA grouped cryptocurrencies results saved to: {filepath}")

def save_anova_post_hoc_grouped(anova_results: Dict[str, any], filename: str = "analysis_anova_post_hoc_grouped.txt") -> None:
    """Salva os resultados do teste post hoc para grupos de criptomoedas em um arquivo .txt.

    Args:
        anova_results (Dict[str, any]): Resultados da ANOVA para grupos de criptomoedas.
        filename (str): Nome do arquivo .txt a ser salvo.

    Returns:
        None
    """
    if not anova_results['is_significant'] or not anova_results['post_hoc']:
        logging.info("Sem resultados significativos para teste post hoc - grupos de criptomoedas")
        return
    
    filepath = PROCESSED_DATA / filename
    
    with open(filepath, mode="w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write("ANÁLISE POST HOC - GRUPOS DE CRIPTOMOEDAS\n")
        file.write("=" * 80 + "\n\n")
        
        file.write(f"Critério de Agrupamento: {anova_results['grouping_criterion']}\n")
        file.write(f"Método do Teste: Teste t pareado\n")
        file.write(f"Objetivo: Identificar quais grupos diferem significativamente entre si\n\n")
        
        file.write("GRUPOS FORMADOS:\n")
        file.write("-" * 40 + "\n")
        for group_name, symbols in anova_results['groups'].items():
            file.write(f"{group_name}: {', '.join(symbols)}\n")
        file.write("\n")
        
        file.write("COMPARAÇÕES PAREADAS:\n")
        file.write("-" * 40 + "\n")
        
        for i, comparison in enumerate(anova_results['post_hoc']['pairwise_comparisons'], 1):
            file.write(f"COMPARAÇÃO {i}: {comparison['group1']} vs {comparison['group2']}\n")
            file.write("-" * 60 + "\n")
            file.write(f"Diferença de Média: {comparison['mean_diff']:.4f}\n")
            file.write(f"Estatística t: {comparison['t_statistic']:.4f}\n")
            file.write(f"Valor p: {comparison['p_value']:.4f}\n")
            file.write(f"Resultado: {'SIGNIFICATIVO' if comparison['significant'] else 'NÃO SIGNIFICATIVO'}\n")
            
            if comparison['significant']:
                if comparison['mean_diff'] > 0:
                    file.write(f"Interpretação: {comparison['group1']} tem retorno médio MAIOR que {comparison['group2']}\n")
                else:
                    file.write(f"Interpretação: {comparison['group1']} tem retorno médio MENOR que {comparison['group2']}\n")
            else:
                file.write(f"Interpretação: Não há diferença significativa entre {comparison['group1']} e {comparison['group2']}\n")
            
            file.write("\n")
        
        file.write("=" * 80 + "\n")
        file.write("RESUMO DA ANÁLISE POST HOC:\n")
        file.write("=" * 80 + "\n")
        
        significant_comparisons = [comp for comp in anova_results['post_hoc']['pairwise_comparisons'] if comp['significant']]
        total_comparisons = len(anova_results['post_hoc']['pairwise_comparisons'])
        
        file.write(f"Total de comparações realizadas: {total_comparisons}\n")
        file.write(f"Comparações significativas: {len(significant_comparisons)}\n")
        file.write(f"Comparações não significativas: {total_comparisons - len(significant_comparisons)}\n\n")
        
        if significant_comparisons:
            file.write("DIFERENÇAS SIGNIFICATIVAS ENCONTRADAS:\n")
            file.write("-" * 50 + "\n")
            for comp in significant_comparisons:
                direction = ">" if comp['mean_diff'] > 0 else "<"
                file.write(f"• {comp['group1']} {direction} {comp['group2']} (p = {comp['p_value']:.4f})\n")
        else:
            file.write("Nenhuma diferença significativa foi encontrada entre os grupos.\n")
        
        file.write("\n" + "=" * 80 + "\n")
        file.write("Análise concluída com sucesso\n")
        file.write("=" * 80 + "\n")
    
    logging.info(f"Post hoc grouped cryptocurrencies results saved to: {filepath}")

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

    # Análise ANOVA - Teste entre criptomoedas
    anova_results_cryptos = anova_analysis_cryptocurrencies(data_filtered, significance_level)
    
    logging.info("Resultados da Análise ANOVA - Criptomoedas Individuais:")
    logging.info(f"  Estatística F: {anova_results_cryptos['anova_f_statistic']:.4f}")
    logging.info(f"  p-valor: {anova_results_cryptos['anova_p_value']:.4f}")
    logging.info(f"  Conclusão: {anova_results_cryptos['anova_conclusion']}")
    
    if anova_results_cryptos["is_significant"]:
        logging.info("  Comparações par a par (Pairwise t-test):")
        for comparison in anova_results_cryptos["post_hoc"]["pairwise_comparisons"]:
            logging.info(f"    {comparison['group1']} vs {comparison['group2']}:")
            logging.info(f"      Diferença de Média: {comparison['mean_diff']:.4f}")
            logging.info(f"      p-valor: {comparison['p_value']:.4f}")
            logging.info(f"      Estatística t: {comparison['t_statistic']:.4f}")
            logging.info(f"      Significativo: {'Sim' if comparison['significant'] else 'Não'}")
    
    logging.info("=" * 70)

    # Análise ANOVA - Teste entre grupos de criptomoedas
    anova_results_groups = anova_analysis_grouped_cryptocurrencies(data_filtered, grouping_criterion="volatility", significance_level=significance_level)
    
    logging.info("Resultados da Análise ANOVA - Grupos de Criptomoedas (Volatilidade):")
    logging.info(f"  Estatística F: {anova_results_groups['anova_f_statistic']:.4f}")
    logging.info(f"  p-valor: {anova_results_groups['anova_p_value']:.4f}")
    logging.info(f"  Conclusão: {anova_results_groups['anova_conclusion']}")
    
    if anova_results_groups["is_significant"]:
        logging.info("  Comparações par a par (Pairwise t-test):")
        for comparison in anova_results_groups["post_hoc"]["pairwise_comparisons"]:
            logging.info(f"    {comparison['group1']} vs {comparison['group2']}:")
            logging.info(f"      Diferença de Média: {comparison['mean_diff']:.4f}")
            logging.info(f"      p-valor: {comparison['p_value']:.4f}")
            logging.info(f"      Estatística t: {comparison['t_statistic']:.4f}")
            logging.info(f"      Significativo: {'Sim' if comparison['significant'] else 'Não'}")
    
    logging.info("=" * 70)
    
    # Salva resultados da análise ANOVA em arquivos CSV separados
    save_anova_individual_cryptocurrencies(anova_results_cryptos)
    save_anova_post_hoc_individual(anova_results_cryptos)
    save_anova_grouped_cryptocurrencies(anova_results_groups)
    save_anova_post_hoc_grouped(anova_results_groups)

if __name__ == "__main__":
    main()