# Projeto Final - Módulo I: Previsão de Preços de Criptomoedas

**Especialização em Inteligência Artificial Aplicada**
**Instituição: Instituto Federal Goiás (IFG)**

**Professores:** Dr. Eduardo Noronha, Me. Otávio Calaça, Dr. Eder Brito
**Data de Entrega:** 10/Jul/2025

```
✅: Tarefas concluídas.
💭: Tarefas a serem pensadas/planejadas.
❓: Tarefas com dúvidas/pendentes.
⏳: Tarefas em andamento/pendentes.
➕: Tarefas concluídas e tarefas a serem adicionadas.
```

## ✅ 1. Visão Geral do Projeto

Este projeto foca no desenvolvimento de um modelo de previsão para o preço de fechamento de criptomoedas. A principal abordagem exigida é o uso de uma rede neural Multi-Layer Perceptron (MLP), com uma análise comparativa em relação a modelos de regressão linear e polinomial.

O projeto abrange desde a coleta e análise estatística dos dados até o treinamento, validação e avaliação de performance dos modelos, incluindo uma análise de rentabilidade de uma estratégia de investimento simulada.

## ✅ 2. Fonte de Dados

Os conjuntos de dados utilizados neste projeto devem ser obtidos do portal CryptoDataDownload:

* URL: [https://www.cryptodatadownload.com](https://www.cryptodatadownload.com)
* Exemplo de Exchange: Dados da Poloniex ([https://www.cryptodatadownload.com/data/poloniex/](https://www.cryptodatadownload.com/data/poloniex/))

As análises estatísticas foram realizadas nas seguintes criptomoedas:
BTC, ETH, LTC, XRP, BCH, XMR, DASH, ETC, BAT, ZRX.

## ✅ 3. Estrutura do Diretório

O projeto está organizado na seguinte estrutura para garantir modularidade e boas práticas de desenvolvimento:

```
/projeto_cripto/
│
├── data/
│   ├── raw/          # Armazena os arquivos .csv brutos
│   └── processed/    # Armazena os dados após limpeza e tratamento
│
├── figures/          # Gráficos e visualizações geradas (mínimo 150 dpi)
│
├── src/
│   ├── __init__.py
│   ├── data_load.py  # Módulo para carregamento dos dados
│   ├── features.py   # Módulo para engenharia de features
│   ├── models.py     # Módulo para treinamento dos modelos
│   ├── analysis.py   # Módulo para análises estatísticas e financeiras
│   └── main.py       # Script principal executável via CLI
│
├── tests/            # Diretório de testes automatizados
│   └── test_*.py     # Arquivos de teste para pytest
│
├── .gitignore        # Arquivos e pastas a serem ignorados pelo Git
├── requirements.txt  # Dependências do projeto Python
└── README.md         # Este arquivo
```

## ✅ 4. Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

Clone o repositório:

```bash
git git@github.com:ronenfilho/cripto_analytics.git
cd projeto_cripto
```

Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

Instale as dependências:
O projeto utiliza ferramentas de lint e formatação como `black` e `ruff`. Instale todas as dependências listadas no arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 💭 5. Como Executar o Projeto

### 5.1. Executando a Análise e Treinamento

O script principal `main.py` foi projetado para ser configurável e executado via linha de comando (CLI) utilizando `argparse`.

Exemplo de uso:

```bash
python src/main.py --crypto BTC --model MLP --kfolds 5
```

Parâmetros disponíveis:

* `--crypto`: Define a sigla da criptomoeda a ser analisada (ex: BTC, ETH).
* `--model`: Especifica o modelo a ser treinado (ex: MLP, Linear, Polynomial).
* `--kfolds`: Número de divisões (folds) para a validação cruzada (K-Fold Cross Validation).

### 5.2. Executando os Testes Automatizados

O projeto inclui testes automatizados para garantir a qualidade e o funcionamento correto dos módulos. Para executá-los, utilize o `pytest`.

Para rodar os testes e gerar um relatório de cobertura:

```bash
pytest --cov=src --cov-report=html
```

## Testes Automatizados

Este projeto utiliza o `pytest` para executar testes automatizados e gerar relatórios de cobertura de código.

### Executando os Testes

1. Certifique-se de que o ambiente virtual está ativado.
2. No diretório raiz do projeto, execute o seguinte comando:

   ```bash
   pytest --cov=src --cov-report=html
   ```

### Relatório de Cobertura

Após a execução dos testes, um relatório de cobertura será gerado no diretório `htmlcov`. Para visualizar o relatório, abra o arquivo `htmlcov/index.html` em um navegador.

## 💭 6. Funcionalidades e Análises Implementadas

### ✅ Análise Estatística

* ✅ Cálculo de medidas resumo e de dispersão para as criptomoedas.
* ✅ Geração de boxplots e histogramas dos preços de fechamento.
* ✅ Analisar a variabilidade entre as criptomoedas com base nas medidas de dispersão.
* ✅ Construir gráfico de linha com o preço de fechamento destacando a média, mediana e moda ao longo do tempo.

### Engenharia de Features

* Seleção e criação de variáveis a partir de dados da própria série (médias móveis, desvio padrão) ou de fontes externas.

### Modelagem e Validação

* Implementação de MLP e modelos de regressão (Linear e Polinomial graus 2 a 10).
* Aplicação da estratégia de validação cruzada K-Fold.

### Análise de Performance e Lucro

* Cálculo do lucro obtido a partir de um investimento inicial de U\$ 1.000,00.
* Comparação dos modelos com base no erro padrão e diagramas de dispersão.
* Gráfico da evolução do lucro para cada modelo.

### Análise de Hipóteses e Variância

* Teste de hipótese para o retorno médio esperado.
* Análise de Variância (ANOVA) para comparar os retornos médios diários entre as criptomoedas e entre grupos de criptomoedas, com testes post hoc quando necessário.

## Uso do Script via Linha de Comando (CLI)

O script `main.py` suporta execução via linha de comando utilizando o módulo `argparse`. Abaixo estão os argumentos disponíveis e exemplos de uso.

### Argumentos Disponíveis

- `--data`: Executa apenas a etapa de carregamento de dados.
- `--features`: Executa apenas a etapa de cálculo de features.
- `--model`: Executa apenas a etapa de treinamento de modelos.
- `--simulate`: Executa apenas a etapa de simulação de investimentos.
- `--crypto`: Especifica os símbolos das criptomoedas a serem analisadas. Exemplo: `BTC/USDT,ETH/USDT`.
- `--kfolds`: Define o número de folds para validação cruzada. Exemplo: `--kfolds 5`.
- `--days`: Define o número de dias para o período de teste na simulação. Exemplo: `--days 30`.
- `--initial_capital`: Define o capital inicial para a simulação de investimento. Exemplo: `--initial_capital 1000.0`.

### Exemplos de Execução

1. **Executar todas as etapas do pipeline:**
   ```bash
   python src/main.py
   ```

2. **Executar apenas a etapa de cálculo de features:**
   ```bash
   python src/main.py --features --crypto BTC/USDT,ETH/USDT
   ```

3. **Executar o pipeline para criptomoedas específicas:**
   ```bash
   python src/main.py --crypto BTC/USDT,ETH/USDT
   ```

4. **Executar a simulação com parâmetros personalizados:**
   ```bash
   python src/main.py --simulate --crypto BTC/USDT --days 365 --initial_capital 5000.0
   ```

5. **Executar com validação cruzada personalizada:**
   ```bash
   python src/main.py --model --kfolds 10
   ```

## Extras 

### 📦 Criação do requirements.txt

```
pip freeze > requirements.txt
```

- **Gegar versão resumida** 
```
pipreqs . --force
```