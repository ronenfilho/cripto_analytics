# 📊 ORDEM DE LEITURA E ANÁLISE DOS RESULTADOS

## 🎯 Sequência Recomendada

### **1️⃣ ANÁLISE BÁSICA INDIVIDUAL**
**📁 `01_analysis_hypothesis_test_results.csv`**
- **Objetivo**: Verificar se cada combinação (criptomoeda + estratégia) atende ao retorno esperado
- **Tipo**: Teste de hipótese t-test unilateral
- **O que observar**: 
  - Quais combinações rejeitam H0 (retorno < esperado)
  - Quais combinações não rejeitam H0 (retorno ≥ esperado)
  - Performance individual de cada estratégia por criptomoeda

---

### **2️⃣ COMPARAÇÃO ENTRE TODAS AS COMBINAÇÕES**
**📁 `02_analysis_anova_individual_cryptos.txt`**
- **Objetivo**: Verificar se há diferenças significativas entre TODAS as combinações
- **Tipo**: ANOVA one-way
- **O que observar**:
  - Se existe diferença estatística entre as combinações (símbolo + estratégia)
  - Estatística F e p-valor geral

**📁 `03_analysis_anova_post_hoc_individual.csv`**
- **Objetivo**: Identificar QUAIS combinações diferem entre si (se ANOVA for significativa)
- **Tipo**: Teste t pareado (pairwise)
- **O que observar**:
  - Comparações específicas entre pares de combinações
  - Quais estratégias/criptomoedas são significativamente diferentes

---

### **3️⃣ ANÁLISE AGRUPADA POR VOLATILIDADE**
**📁 `04_analysis_anova_grouped_cryptos_volatility.txt`**
- **Objetivo**: Verificar se a volatilidade das combinações influencia o retorno
- **Tipo**: ANOVA entre grupos (Alta vs Baixa Volatilidade)
- **O que observar**:
  - Se grupos de alta/baixa volatilidade têm retornos diferentes
  - Quais combinações estão em cada grupo

**📁 `05_analysis_anova_post_hoc_grouped_volatility.txt`**
- **Objetivo**: Detalhar diferenças entre grupos de volatilidade (se significativo)
- **Tipo**: Teste t entre grupos
- **O que observar**:
  - Se alta volatilidade resulta em maior/menor retorno
  - Magnitude da diferença entre grupos

---

### **4️⃣ ANÁLISE AGRUPADA POR RETORNO MÉDIO**
**📁 `06_analysis_anova_grouped_cryptos_mean_return.txt`**
- **Objetivo**: Verificar se é possível separar combinações em grupos de performance
- **Tipo**: ANOVA entre grupos (Alto vs Baixo Retorno)
- **O que observar**:
  - Se a classificação por retorno médio é estatisticamente válida
  - Quais combinações estão no grupo de alto/baixo desempenho

**📁 `07_analysis_anova_post_hoc_grouped_mean_return.txt`**
- **Objetivo**: Confirmar e quantificar diferenças entre grupos de performance
- **Tipo**: Teste t entre grupos
- **O que observar**:
  - Magnitude da diferença entre grupos de alto/baixo retorno
  - Significância estatística da separação

---

## 🔍 **ROTEIRO DE ANÁLISE**

### **Passo 1: Análise Individual** (Arquivos 01-03)
1. Abrir `01_analysis_hypothesis_test_results.csv`
2. Identificar melhores e piores combinações
3. Verificar `02_analysis_anova_individual_cryptos.txt` para significância geral
4. Se significativo, analisar `03_analysis_anova_post_hoc_individual.csv` para detalhes

### **Passo 2: Análise por Agrupamento** (Arquivos 04-07)
1. Comparar `04_analysis_anova_grouped_cryptos_volatility.txt` vs `06_analysis_anova_grouped_cryptos_mean_return.txt`
2. Verificar qual critério (volatilidade ou retorno médio) é mais discriminante
3. Analisar os respectivos post hoc (05 e 07) para entender as diferenças

### **Passo 3: Síntese**
1. Integrar insights de análises individuais e agrupadas
2. Identificar padrões consistentes
3. Formular conclusões sobre estratégias e criptomoedas

---

## 📈 **INTERPRETAÇÃO DOS RESULTADOS**

### **P-valor < 0.05**: Significativo
- Há evidência estatística de diferença
- Rejeitar hipótese nula

### **P-valor ≥ 0.05**: Não significativo
- Não há evidência suficiente de diferença
- Não rejeitar hipótese nula

### **Estatística F alta**: Maior variabilidade entre grupos
### **Estatística t**: Direção e magnitude da diferença

---

*📝 Documentação gerada automaticamente pelo sistema de análise cripto_analytics*
