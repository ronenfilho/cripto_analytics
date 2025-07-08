import pandas as pd
import os
from scipy.stats import ttest_1samp
from pathlib import Path

# Importa o caminho do arquivo de configuração
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA, EXPECTED_RETURN, SIGNIFICANCE_LEVEL

def load_simulation_results():
    """
    Carrega os dados de retorno do arquivo simulation_results_days.csv
    
    Returns:
        pd.DataFrame: DataFrame com os dados de simulação
    """
    file_path = PROCESSED_DATA / "simulation_results_days.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    return pd.read_csv(file_path)

def hypothesis_test_by_strategy(data: pd.DataFrame, expected_return: float, significance_level: float = 0.05):
    """
    Realiza um teste de hipótese para verificar se o retorno médio esperado é superior ou igual a um valor definido pelo usuário,
    considerando cada combinação de símbolo e estratégia.

    H₀ (Hipótese Nula): O retorno médio diário é maior ou igual a x%. (μ ≥ x)
    H₁ (Hipótese Alternativa): O retorno médio diário é menor que x%. (μ < x)

    Args:
        data (pd.DataFrame): DataFrame contendo as colunas ['symbol', 'strategy', 'return_pct'].
        expected_return (float): Retorno esperado médio (em porcentagem).
        significance_level (float): Nível de significância para o teste (default: 0.05).

    Returns:
        dict: Resultado do teste para cada combinação de símbolo e estratégia, incluindo o p-valor e a conclusão.
    """
    results = {}

    for (symbol, strategy), group in data.groupby(["symbol", "strategy"]):
        returns = group["return_pct"].dropna()

        # Realiza o teste t de uma amostra (teste unilateral à esquerda)
        # H₀: μ ≥ expected_return (retorno médio é maior ou igual ao esperado)
        # H₁: μ < expected_return (retorno médio é menor que o esperado)
        t_stat, p_value_bilateral = ttest_1samp(returns, expected_return / 100)
        
        # Para teste unilateral à esquerda (μ < x), usamos:
        # Se t_stat < 0: p_value = p_value_bilateral / 2
        # Se t_stat ≥ 0: p_value = 1 - (p_value_bilateral / 2)
        if t_stat < 0:
            p_value = p_value_bilateral / 2
        else:
            p_value = 1 - (p_value_bilateral / 2)

        # Conclusão do teste unilateral
        if t_stat < 0 and p_value < significance_level:
            conclusion = "Rejeitar H₀ - O retorno médio é significativamente menor que o esperado"
        else:
            conclusion = "Não rejeitar H₀ - O retorno médio é maior ou igual ao esperado"

        results[(symbol, strategy)] = {
            "t_stat": t_stat,
            "p_value": p_value,
            "conclusion": conclusion,
        }

    return results

if __name__ == "__main__":
    # Carrega os dados de simulação
    data = load_simulation_results()
    
    # Filtra apenas os dados onde o investimento foi realizado
    data_filtered = data[data['investment_made'] == 'Yes']
    
    expected_return = EXPECTED_RETURN
    significance_level = SIGNIFICANCE_LEVEL

    print(f"Análise Estatística - Teste de Hipótese")
    print(f"Retorno Esperado: {expected_return:.2f}%")
    print(f"Nível de Significância: {significance_level:.2f}")
    print(f"Total de registros: {len(data)}")
    print(f"Registros com investimento realizado: {len(data_filtered)}")
    print("=" * 70)
    
    print("\nDESCRIÇÃO DO TESTE DE HIPÓTESE:")
    print("H₀ (Hipótese Nula): O retorno médio diário é maior ou igual ao valor esperado (μ ≥ x%)")
    print("H₁ (Hipótese Alternativa): O retorno médio diário é menor que o valor esperado (μ < x%)")
    print(f"Teste: t-test unilateral à esquerda (μ < {expected_return:.2f}%)")
    print("Interpretação:")
    print(f"  - p-valor < {significance_level:.2f}: Rejeitar H₀ (retorno é significativamente menor que o esperado)")
    print(f"  - p-valor ≥ {significance_level:.2f}: Não rejeitar H₀ (retorno é maior ou igual ao esperado)")
    print("  - Estatística t < 0: Retorno observado < esperado")
    print("  - Estatística t ≥ 0: Retorno observado ≥ esperado")
    print("=" * 70)

    # Teste por símbolo e estratégia
    results_by_strategy = hypothesis_test_by_strategy(data_filtered, expected_return, significance_level)
    
    print("\nResultados por símbolo e estratégia:")
    for (symbol, strategy), result in results_by_strategy.items():
        # Calcula a média alcançada para este grupo
        group_data = data_filtered[(data_filtered['symbol'] == symbol) & (data_filtered['strategy'] == strategy)]
        achieved_mean = group_data['return_pct'].mean() * 100  # Converte para porcentagem

        print(f"Criptomoeda: {symbol}, Estratégia: {strategy}")
        print(f"  Média Esperada: {expected_return:.4f}%")
        print(f"  Média Alcançada: {achieved_mean:.4f}%")
        print(f"  Estatística t: {result['t_stat']:.4f}")
        print(f"  p-valor: {result['p_value']:.4f}")
        print(f"  Conclusão: {result['conclusion']}")
        print("-" * 40)