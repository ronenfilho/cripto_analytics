# 🚀 Projeto Final - Módulo I: Previsão de Preços de Criptomoedas

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style](https://img.shields.io/badge/code%20style-ruff%20%7C%20black-000000.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)

**Instituição: Instituto Federal Goiás (IFG)**<br>
**Especialização em Inteligência Artificial Aplicada**

**Professores:** Dr. Eder Brito, Dr. Eduardo Noronha e Me. Otávio Calaça
**Data de Entrega:** 10/Jul/2025

## 📋 Índice

- [🎯 Início Rápido](#-início-rápido)
- [⚙️ Configurações](#-configurações)
- [✅ Visão Geral do Projeto](#-1-visão-geral-do-projeto)
- [📄 Fonte de Dados](#-2-fonte-de-dados)
- [🏗️ Estrutura do Diretório](#-3-estrutura-do-diretório)
- [⚙️ Instalação e Configuração](#-4-instalação-e-configuração)
- [🚀 Como Executar o Projeto](#-5-como-executar-o-projeto)
- [💡 Funcionalidades Implementadas](#-6-funcionalidades-e-análises-implementadas)
- [📊 Análise de Resultados](#-7-análise-dos-resultados)
- [⚡ Recursos Avançados](#-8-recursos-avançados)
- [🧪 Testes Automatizados](#-testes-automatizados)
- [⚠️ Troubleshooting](#-troubleshooting)
- [❓ FAQ (Perguntas Frequentes)](#-faq-perguntas-frequentes)
- [🤝 Contribuição](#-contribuição)
- [📝 Changelog](#-changelog)

## 🎯 Início Rápido

```powershell
# Clone e configure o projeto
git clone git@github.com:ronenfilho/cripto_analytics.git
cd cripto_analytics

# Configure o ambiente
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Para desenvolvimento, instale as dependências adicionais
pip install -r requirements-dev.txt

# Configure as variáveis
copy src\.env.example src\.env

# Execute o pipeline completo
python src\main.py
```

> 📊 **Resultados**: Após a execução, verifique:
> - Logs de execução em `app.log`
> - Gráficos no diretório `figures\`
> - Resultados no diretório `data\processed\`
> - Guia de análise em [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md)

---

## 📊 Análise de Resultados

Para interpretação dos resultados estatísticos e de performance, consulte o guia detalhado: [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md)

Este guia fornece:
- Explicação dos arquivos gerados (01-07)
- Interpretação das análises estatísticas
- Comparação de desempenho entre modelos
- Conclusões sobre os resultados da simulação

---



## ⚙️ Configurações

Para personalizar o comportamento do projeto, consulte a documentação detalhada de configurações:
- **Guia Completo**: [`docs/CONFIGURACOES.md`](docs/CONFIGURACOES.md)
- **Arquivo de Exemplo**: [`src\.env.example`](src/.env.example)

As configurações permitem personalizar:
- Símbolos de criptomoedas analisados
- Modelos de machine learning utilizados
- Parâmetros de simulação
- Configurações de visualização
- Níveis de log e execução



## ✅ 1. Visão Geral do Projeto

Este projeto foca no desenvolvimento de um modelo de previsão para o preço de fechamento de criptomoedas. A principal abordagem exigida é o uso de uma rede neural Multi-Layer Perceptron (MLP), com uma análise comparativa em relação a modelos de regressão linear e polinomial.

O projeto abrange desde a coleta e análise estatística dos dados até o treinamento, validação e avaliação de performance dos modelos, incluindo uma análise de rentabilidade de uma estratégia de investimento simulada.

## ✅ 2. Fonte de Dados

Os conjuntos de dados utilizados neste projeto devem ser obtidos do portal CryptoDataDownload:

* URL: [https://www.cryptodatadownload.com](https://www.cryptodatadownload.com)
* Exemplo de Exchange: Dados da Poloniex ([https://www.cryptodatadownload.com/data/poloniex/](https://www.cryptodatadownload.com/data/poloniex/))

As análises estatísticas foram realizadas nas seguintes criptomoedas:
**BCH/USDT, BTC/USDT, DASH/USDT, EOS/USDT, ETC/USDT, ETH/USDT, LTC/USDT, XMR/USDT, XRP/USDT, ZRX/USDT**

> 📊 **Dados**: Total de 10 criptomoedas com dados históricos diários da exchange Poloniex

## ✅ 3. Estrutura do Diretório

O projeto está organizado na seguinte estrutura para garantir modularidade e boas práticas de desenvolvimento:

```
/cripto_analytics/
│
├── data/
│   ├── raw/                    # Dados brutos das criptomoedas (Poloniex CSV)
│   │   ├── Poloniex_BCHUSDT_d.csv
│   │   ├── Poloniex_BTCUSDT_d.csv
│   │   ├── Poloniex_DASHUSDT_d.csv
│   │   ├── Poloniex_EOSUSDT_d.csv
│   │   ├── Poloniex_ETCUSDT_d.csv
│   │   ├── Poloniex_ETHUSDT_d.csv
│   │   ├── Poloniex_LTCUSDT_d.csv
│   │   ├── Poloniex_XMRUSDT_d.csv
│   │   ├── Poloniex_XRPUSDT_d.csv
│   │   └── Poloniex_ZRXUSDT_d.csv
│   └── processed/              # Dados processados e resultados de análises
│       ├── 00_ORDEM_DE_LEITURA_E_ANALISE.md  # 📖 Guia de análise dos resultados
│       ├── 01_analysis_hypothesis_test_results.csv
│       ├── 02_analysis_anova_individual_cryptos.txt
│       ├── 03_analysis_anova_post_hoc_individual.csv
│       ├── 04_analysis_anova_grouped_cryptos_volatility.txt
│       ├── 05_analysis_anova_post_hoc_grouped_volatility.txt
│       ├── 06_analysis_anova_grouped_cryptos_mean_return.txt
│       ├── 07_analysis_anova_post_hoc_grouped_mean_return.txt
│       ├── simulation_results_days.csv
│       └── simulation_results_consolidated.csv
│
├── figures/                    # Gráficos e visualizações (PNG, mínimo 150 dpi)
│
├── src\                        # Código fonte principal
│   ├── __init__.py
│   ├── .env                    # Variáveis de ambiente e configurações
│   ├── .env.example            # Template de configurações
│   ├── config.py               # Configurações globais do projeto
│   ├── data_load.py            # Carregamento e limpeza dos dados
│   ├── features.py             # Engenharia de features e preprocessamento
│   ├── models.py               # Implementação dos modelos ML (MLP, Linear, Polynomial)
│   ├── analysis.py             # Análises estatísticas e testes de hipótese
│   ├── simulate.py             # Simulação de investimentos e estratégias
│   ├── plot.py                 # Geração de gráficos e visualizações
│   ├── utils.py                # Funções utilitárias e helpers
│   └── main.py                 # Script principal executável via CLI
│
├── tests/                      # Testes automatizados (pytest)
│   ├── conftest.py             # Configurações e fixtures para testes
│   ├── test_data_load.py       # Testes do módulo data_load
│   ├── test_features.py        # Testes do módulo features
│   ├── test_main.py            # Testes do script principal
│   ├── test_models.py          # Testes dos modelos ML
│   └── test_simulate.py        # Testes da simulação de investimentos
│
├── docs/                       # Documentação do projeto
│   ├── CONFIGURACOES.md        # Documentação detalhada das configurações
│   └── Trabalho IA - Final (2025).pdf
│
├── htmlcov/                    # Relatórios de cobertura de testes (gerado)
├── venv2/                      # Ambiente virtual Python (local)
├── .coverage                   # Arquivo de cobertura de testes
├── .gitignore                  # Arquivos ignorados pelo Git
├── .pytest_cache/              # Cache do pytest
├── .ruff_cache/                # Cache do linter Ruff
├── app.log                     # Logs da aplicação
├── pyproject.toml              # Configurações do projeto Python (Ruff, etc.)
├── pytest.ini                 # Configurações do pytest
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## ✅ 4. Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

Clone o repositório:

```powershell
git clone git@github.com:ronenfilho/cripto_analytics.git
cd cripto_analytics
```

Crie e ative um ambiente virtual (recomendado):

```powershell
python -m venv venv
venv\Scripts\activate  # No Windows
# No Linux/Mac: source venv/bin/activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

### Dependências Principais

| Biblioteca | Versão | Propósito |
|------------|---------|-----------|
| `pandas` | 2.3.0 | Manipulação de dados |
| `numpy` | 2.3.1 | Computação numérica |
| `scikit-learn` | 1.7.0 | Machine Learning |
| `matplotlib` | 3.10.3 | Visualizações |
| `python-dotenv` | 1.1.1 | Configurações .env |

> 📋 **Nota**: Outras dependências para desenvolvimento incluem `pytest`, `black`, `ruff` (instale com `pip install -r requirements-dev.txt`).

### 4.2. Configuração do Projeto

**📝 Copie o arquivo de configuração:**
```powershell
copy src\.env.example src\.env
```

**⚙️ Personalize as configurações no arquivo `src\.env`:**
- Símbolos de criptomoedas para análise
- Modelos de ML a serem utilizados
- Parâmetros de simulação
- Configurações de visualização

> 📚 **Documentação completa**: Consulte [`docs/CONFIGURACOES.md`](docs/CONFIGURACOES.md) para detalhes sobre cada configuração.

## 💭 5. Como Executar o Projeto

### 5.1. Pipeline Completo

**🚀 Execução completa (recomendado para primeira execução):**
```powershell
python src\main.py
```

### 5.2. Execuções Específicas

**🎯 Simulação de investimento:**
```powershell
python src\simulate.py
```

**📊 Apenas análise estatística:**
```powershell
python src\analysis.py
```

### 5.3. Parâmetros Personalizados

**🔧 Pipeline com criptomoedas específicas:**
```powershell
python src\main.py --crypto BTC/USDT,ETH/USDT
```

**💰 Simulação personalizada:**
```powershell
python src\main.py --simulate --crypto BTC/USDT --days 30 --initial_capital 5000.0
```

**🧠 Modelos específicos:**
```powershell
python src\main.py --model --kfolds 10
```

**💡 Use o help para ver todas as opções:**
```powershell
python src\main.py --help
```

> ⚙️ **Dica**: As configurações padrão estão no arquivo `src\.env` e podem ser personalizadas conforme necessário.

### 5.4. Exemplo de Saída

Ao executar `python src\main.py`, você verá uma saída similar a:

```
INFO - Iniciando pipeline de análise de criptomoedas...
INFO - Carregando dados para 10 símbolos...
INFO - Processando features para BTC/USDT...
INFO - Treinando modelos: Linear, MLP, Polynomial...
INFO - Executando simulação de investimentos...
INFO - Gerando visualizações...
INFO - Salvando resultados em data/processed/...
INFO - Pipeline concluído com sucesso! 
```

**📁 Arquivos gerados:**
- `data\processed\01_analysis_hypothesis_test_results.csv`
- `data\processed\simulation_results_consolidated.csv`
- `figures\*_analysis.png`

## 🧪 Testes Automatizados

Este projeto utiliza o `pytest` para executar testes automatizados e gerar relatórios de cobertura de código.

**🧪 Executar todos os testes:**
```powershell
pytest
```

**📋 Testes com relatório de cobertura:**
```powershell
pytest --cov=src --cov-report=html
```

**📊 Visualizar relatório de cobertura:**
O relatório HTML será gerado em `htmlcov\index.html`

> 🔍 **Dica**: Use `pytest -v` para saída detalhada dos testes.

### Executando a Análise Estatística

Para executar as análises estatísticas completas (testes de hipótese, ANOVA, comparações de grupos):

```powershell
python src\analysis.py
```

> 📊 **Importante**: Após a execução, consulte o guia [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md) para interpretar os resultados de forma estruturada.

## 💭 6. Funcionalidades e Análises Implementadas

### ✅ Análise Estatística Completa

* ✅ **Medidas resumo e dispersão** para todas as criptomoedas
* ✅ **Visualizações**: Boxplots, histogramas, gráficos de linha temporais
* ✅ **Análise de variabilidade** entre criptomoedas
* ✅ **Estatísticas descritivas** com média, mediana, moda e tendências

### ✅ Engenharia de Features

* ✅ **Médias móveis** (diferentes períodos)
* ✅ **Indicadores técnicos** (RSI, MACD, Bollinger Bands)
* ✅ **Features temporais** (dia da semana, mês, tendências)
* ✅ **Variáveis de volatilidade** e momentum

### ✅ Modelagem e Validação

* ✅ **Modelos implementados**: MLP, Regressão Linear, Regressão Polinomial (graus 2-10)
* ✅ **Validação cruzada** K-Fold configurável
* ✅ **Métricas de avaliação**: RMSE, MSE
* ✅ **Comparação automática** de performance entre modelos

### ✅ Simulação de Investimento

* ✅ **Capital inicial configurável** (padrão: $1000 USD)
* ✅ **Estratégias de investimento** baseadas em previsões
* ✅ **Análise de retorno** diário e consolidado
* ✅ **Comparação de performance** entre modelos e estratégias
* ✅ **Visualizações** da evolução do capital

### ✅ Análise Estatística Avançada

* ✅ **Testes de hipótese** para retorno esperado
* ✅ **ANOVA** para comparação entre grupos
* ✅ **Análise post hoc** para diferenças específicas
* ✅ **Agrupamento por critérios**: volatilidade, retorno médio
* ✅ **Relatórios estruturados** em formatos CSV e TXT

## 📊 7. Análise dos Resultados

### 📖 Guia de Leitura dos Resultados

Para uma análise completa e estruturada dos resultados estatísticos e de performance, consulte o guia detalhado:

**📁 [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md)**

Este arquivo contém:
- ✅ **Ordem sequencial de leitura** dos arquivos de resultados (01 a 07)
- ✅ **Explicação detalhada** do objetivo de cada análise
- ✅ **Roteiro passo-a-passo** para interpretação dos dados
- ✅ **Critérios de significância** estatística
- ✅ **Guia de interpretação** dos p-valores e estatísticas

### 📁 Arquivos de Resultados Gerados

Após executar `python src\analysis.py`, os seguintes arquivos serão criados em `data\processed\`:

1. **01_analysis_hypothesis_test_results.csv** - Teste de hipóteses individual
2. **02_analysis_anova_individual_cryptos.txt** - ANOVA entre todas as combinações
3. **03_analysis_anova_post_hoc_individual.csv** - Comparações pareadas detalhadas
4. **04_analysis_anova_grouped_cryptos_volatility.txt** - Análise por volatilidade
5. **05_analysis_anova_post_hoc_grouped_volatility.txt** - Post hoc volatilidade
6. **06_analysis_anova_grouped_cryptos_mean_return.txt** - Análise por retorno médio
7. **07_analysis_anova_post_hoc_grouped_mean_return.txt** - Post hoc retorno médio

> 💡 **Dica**: Sempre comece lendo o arquivo `00_ORDEM_DE_LEITURA_E_ANALISE.md` para entender a sequência lógica de análise.

## 🚀 8. Recursos Avançados

### ⚙️ Configuração Flexível

**📁 Arquivo de configuração**: `src\.env`
- ✅ **Símbolos personalizáveis** para análise e simulação
- ✅ **Modelos ativados/desativados** individualmente
- ✅ **Parâmetros de simulação** ajustáveis
- ✅ **Configurações de visualização** (DPI, formato, tamanho)
- ✅ **Níveis de logging** configuráveis

**📚 Documentação**: [`docs/CONFIGURACOES.md`](docs/CONFIGURACOES.md)

### 📊 Argumentos CLI Disponíveis

**Execução seletiva:**
- `--data`: Apenas carregamento de dados
- `--features`: Apenas engenharia de features
- `--model`: Apenas treinamento de modelos
- `--simulate`: Apenas simulação de investimentos
- `--analysis.py`: Apenas análises estastisticas (Hipótese / ANOVA)

**Parâmetros personalizados:**
- `--crypto`: Símbolos específicos (ex: `BTC/USDT,ETH/USDT`)
- `--kfolds`: Número de folds para validação cruzada
- `--days`: Período de teste da simulação
- `--initial_capital`: Capital inicial da simulação

### 🧪 Qualidade de Código

**🔍 Testes automatizados:**
- ✅ Cobertura de testes com `pytest`
- ✅ Testes unitários para todos os módulos
- ✅ Relatórios HTML de cobertura

**🛠️ Ferramentas de desenvolvimento:**
- ✅ `ruff` para linting e formatação
- ✅ `black` para formatação de código
- ✅ Configuração em `pyproject.toml`

### 📈 Saídas e Visualizações

**📊 Gráficos gerados:**
- Histogramas e boxplots por criptomoeda
- Gráficos de tendência temporal
- Simulações de investimento
- Análise de variabilidade

**📁 Arquivos de resultados:**
- Resultados estruturados e numerados (01-07)
- Formatos CSV e TXT para fácil análise
- Logs detalhados de execução
- Pasta Figures com todos os gráficos

## 🔧 9. Desenvolvimento e Contribuição

### 📦 Gestão de Dependências

**Criar requirements.txt completo:**
```bash
pip freeze > requirements.txt
```

**Gerar versão minimalista:**
```bash
pipreqs.exe . --force --ignore venv
```

### 🌟 Estrutura de Desenvolvimento

**� Organização modular:**
- Separação clara entre dados, código e resultados
- Configurações centralizadas em `.env`
- Documentação abrangente e atualizada
- Testes automatizados e CI/CD ready (TODO)

---

## 📚 Documentação Adicional

- **📊 Análise de Resultados**: [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md)
- **⚙️ Configurações**: [`docs/CONFIGURACOES.md`](docs/CONFIGURACOES.md)
- **📋 Trabalho Acadêmico**: [`docs/Trabalho IA - Final (2025).pdf`](docs/Trabalho%20IA%20-%20Final%20(2025).pdf)
- **📝 Histórico de Alterações**: [docs/CHANGELOG.md](docs/CHANGELOG.md)


### 🌟 Estrutura de Desenvolvimento

**📦 Organização modular:**
- Separação clara entre dados, código e resultados
- Configurações centralizadas em `.env`
- Documentação abrangente e atualizada
- Testes automatizados e CI/CD ready (TODO)

## ⚠️ Troubleshooting

### Problemas Comuns

**❌ Erro: "Module not found"**
```powershell
# Certifique-se de que está no diretório correto e o ambiente virtual ativado
cd cripto_analytics
venv\Scripts\activate
pip install -r requirements.txt
```

**❌ Erro: "No data files found"**
- Verifique se os arquivos CSV estão em `data\raw\`
- Confirme se os nomes dos arquivos seguem o padrão `Poloniex_*USDT_d.csv`

**❌ Erro: "Permission denied" no Windows**
```powershell
# Execute o PowerShell como administrador
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**❌ Gráficos não são gerados**
- Verifique se o diretório `figures\` existe
- Confirme as configurações de DPI e formato no `.env`

### Performance

**⚡ Execução lenta?**
- Reduza o número de símbolos em `SYMBOLS_TO_SIMULATE`
- Diminua o `POLYNOMIAL_DEGREE_RANGE`
- Use menos folds na validação cruzada

**💾 Uso alto de memória?**
- Execute análises por partes usando argumentos específicos
- Monitore o uso com `--log-level DEBUG`

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Crie uma branch** para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)
5. **Abra um Pull Request**

### Padrões de Código

#### Checar e Rodar Black
```powershell
# Checar formatação com black
black --check src\

# Aplicar formatação com black
black src\
```

#### Checar e Rodar Ruff
```powershell
# Checar linting com ruff
ruff check src\

# Aplicar correções automáticas com ruff
ruff check src\ --fix
```

#### Testes antes de commit
```powershell
pytest --cov=src
```

### Estrutura de Commits

- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Atualização de documentação
- `style:` Mudanças de formatação
- `refactor:` Refatoração de código
- `test:` Adição ou modificação de testes

## 📊 Performance Benchmarks

### Tempo de Execução (Médio)

| Operação | Tempo Estimado | CPU/RAM |
|----------|----------------|---------|
| Pipeline Completo | 5-10 min | 4GB RAM |
| Análise Estatística | 2-3 min | 2GB RAM |
| Simulação 10 Símbolos | 3-5 min | 3GB RAM |
| Treinamento MLP | 1-2 min | 1GB RAM |

### Comparação de Modelos

| Modelo | RMSE Médio | Tempo Treino | Complexidade |
|--------|------------|--------------|--------------|
| Linear | 0.085 | 5s | Baixa |
| Polinomial (grau 3) | 0.078 | 15s | Média |
| MLP | 0.072 | 45s | Alta |

## ❓ FAQ (Perguntas Frequentes)

### Uso Geral

**Q: Posso usar outras criptomoedas além das fornecidas?**
A: Sim, mas você precisará baixar os dados no formato correto da Poloniex e ajustar o código de carregamento.

**Q: O projeto funciona em Mac/Linux?**
A: Sim, apenas ajuste os comandos do PowerShell para bash (`python src/main.py` ao invés de `python src\main.py`).

**Q: Quanto tempo demora uma execução completa?**
A: Entre 5-10 minutos para o pipeline completo com todas as 10 criptomoedas.

### Configuração

**Q: Como altero quais modelos são executados?**
A: Edite o arquivo `src\.env` e defina `USE_*_REGRESSION=True/False` para cada modelo.

**Q: Como reduzir o tempo de execução?**
A: Reduza o número de símbolos em `SYMBOLS_TO_SIMULATE` ou diminua `POLYNOMIAL_DEGREE_RANGE`.

**Q: Posso executar apenas a simulação?**
A: Sim, use `python src\simulate.py` ou `python src\main.py --simulate`.

### Resultados

**Q: Como interpreto os resultados estatísticos?**
A: Sempre comece pelo guia [`data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md`](data/processed/00_ORDEM_DE_LEITURA_E_ANALISE.md).

**Q: Onde encontro os gráficos gerados?**
A: No diretório `figures\` após a execução.

**Q: Como salvo configurações personalizadas?**
A: Faça backup do seu arquivo `src\.env` personalizado.

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](docs/LICENSE) para detalhes.

## 👥 Autores

- **Desenvolvedor Principal**: [Ronen Filho](https://github.com/ronenfilho)
- **Orientadores**: Dr. Eduardo Noronha, Me. Otávio Calaça, Dr. Eder Brito
- **Instituição**: Instituto Federal Goiás (IFG)

## 📝 Changelog

Todas as alterações significativas no projeto estão documentadas no arquivo [docs/CHANGELOG.md](docs/CHANGELOG.md).

Versões principais:
- **1.0.0** (10/07/2025): Versão estável com pipeline completo
- **0.9.0** (15/06/2025): Versão inicial do sistema de previsão

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!**