# 📋 DOCUMENTAÇÃO DAS CONFIGURAÇÕES (.env)

## 🎯 Visão Geral

O arquivo `.env` centraliza todas as configurações do projeto Cripto Analytics, permitindo personalização sem modificar o código fonte.

## 📊 Configurações de Dados

### SYMBOLS
- **Descrição**: Lista de criptomoedas para análise geral
- **Formato**: Lista separada por vírgulas
- **Exemplo**: `BTC/USDT,ETH/USDT,LTC/USDT`
- **Padrão**: Todas as 10 criptomoedas disponíveis

### SYMBOLS_TO_SIMULATE
- **Descrição**: Criptomoedas específicas para simulação de investimento
- **Formato**: Lista separada por vírgulas
- **Uso**: Pode ser um subconjunto de SYMBOLS para testes focados
- **Dica**: Use menos símbolos para execução mais rápida

## 🧠 Configurações de Modelos

### USE_LINEAR_REGRESSION / USE_MLP_REGRESSOR / USE_POLYNOMIAL_REGRESSION
- **Descrição**: Ativar/desativar modelos específicos
- **Valores**: `True` ou `False`
- **Impacto**: Controla quais modelos serão treinados e comparados

### POLYNOMIAL_DEGREE_RANGE
- **Descrição**: Intervalo de graus para testar na regressão polinomial
- **Formato**: `min,max` (exemplo: `2,5`)
- **Impacto**: Mais graus = mais modelos testados = maior tempo de execução

## 💰 Configurações de Simulação

### INITIAL_CAPITAL
- **Descrição**: Capital inicial para simulação em USD
- **Formato**: Número decimal
- **Exemplo**: `1000.0` = $1000 USD
- **Impacto**: Base para cálculo de retornos absolutos

### TEST_PERIOD_DAYS
- **Descrição**: Período de teste para simulação
- **Formato**: Número inteiro
- **Exemplo**: `30` = últimos 30 dias dos dados
- **Impacto**: Mais dias = simulação mais longa e robusta

## 📈 Configurações Estatísticas

### EXPECTED_RETURN
- **Descrição**: Retorno esperado médio diário
- **Formato**: Porcentagem decimal
- **Exemplo**: `1.0` = expectativa de 1% ao dia
- **Uso**: Base para testes de hipótese

### SIGNIFICANCE_LEVEL
- **Descrição**: Nível de significância para testes estatísticos
- **Formato**: Decimal entre 0 e 1
- **Valores comuns**: `0.01` (1%), `0.05` (5%), `0.10` (10%)
- **Uso**: Critério para rejeição de hipóteses

## 🎨 Configurações de Visualização

### FIGURE_DPI
- **Descrição**: Resolução dos gráficos gerados
- **Formato**: Número inteiro
- **Recomendado**: `150` (mínimo), `300` (alta qualidade)
- **Impacto**: Maior DPI = arquivos maiores, melhor qualidade

### FIGURE_FORMAT
- **Descrição**: Formato de salvamento dos gráficos
- **Opções**: `png`, `jpg`, `pdf`, `svg`
- **Recomendado**: `png` (boa qualidade/tamanho)

### FIGURE_SIZE_WIDTH / FIGURE_SIZE_HEIGHT
- **Descrição**: Dimensões padrão dos gráficos em polegadas
- **Formato**: Números decimais
- **Exemplo**: `12,8` = 12x8 polegadas
- **Dica**: Ajuste conforme sua tela/apresentação

## 🔧 Configurações de Sistema

### LOG_LEVEL
- **Descrição**: Nível de detalhamento dos logs
- **Opções**: 
  - `DEBUG`: Máximo detalhamento
  - `INFO`: Informações importantes
  - `WARNING`: Apenas avisos e erros
  - `ERROR`: Apenas erros críticos
- **Desenvolvimento**: Use `DEBUG`
- **Produção**: Use `INFO` ou `WARNING`

### USE_TIMING
- **Descrição**: Ativar medição de tempo de execução
- **Valores**: `True` ou `False`
- **Uso**: Útil para otimização de performance

## 📊 Configurações Avançadas

### CROSS_VALIDATION_FOLDS
- **Descrição**: Número de divisões para validação cruzada
- **Formato**: Número inteiro
- **Comum**: `5` ou `10`
- **Impacto**: Mais folds = validação mais robusta, maior tempo

### PRIMARY_METRIC
- **Descrição**: Métrica principal para comparação de modelos
- **Opções**: `mse`, `rmse`, `mae`, `r2`
- **Recomendado**: `rmse` (interpretação mais intuitiva)

### ANOVA_GROUPING_CRITERIA
- **Descrição**: Critérios para agrupamento na análise ANOVA
- **Opções**: `volatility`, `mean_return`, `investment_frequency`
- **Formato**: Lista separada por vírgulas
- **Uso**: Define quais análises de grupo serão executadas

## 🚀 Dicas de Uso

### Para Desenvolvimento/Teste Rápido:
```env
SYMBOLS_TO_SIMULATE=BTC/USDT,ETH/USDT
TEST_PERIOD_DAYS=10
LOG_LEVEL=INFO
USE_TIMING=True
```

### Para Análise Completa:
```env
SYMBOLS_TO_SIMULATE=BCH/USDT,BTC/USDT,DASH/USDT,EOS/USDT,ETC/USDT,ETH/USDT,LTC/USDT,XMR/USDT,XRP/USDT,ZRX/USDT
TEST_PERIOD_DAYS=30
LOG_LEVEL=INFO
FIGURE_DPI=300
```

### Para Depuração:
```env
SYMBOLS_TO_SIMULATE=BTC/USDT
LOG_LEVEL=DEBUG
USE_TIMING=True
```

## ⚠️ Importantes

1. **Nunca commite o arquivo `.env`** (já está no .gitignore)
2. **Use o `.env.example`** como base
3. **Teste configurações** com poucos símbolos primeiro
4. **Backup suas configurações** customizadas
5. **Documente mudanças** específicas do seu ambiente
