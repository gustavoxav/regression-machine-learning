# 📊 Documentação: Pré-processamento de Dados
## Projeto: Regressão de Salários em Data Science
### Dataset: `ds_salaries.csv` | Target: `salary_in_usd`

> **Referência base:** [Mastering Tuning 10 Regressors (>0.5 R²) - Kaggle](https://www.kaggle.com/code/leekahwin/mastering-tuning-10-regressors-0-5-r2)

---

## 1. Descrição do Dataset

| Atributo | Valor |
|---|---------|
| **Nome** | ds_salaries.csv |
| **Linhas originais** | 607 |
| **Colunas originais** | 12 (11 após remoção da coluna índice) |
| **Duplicatas removidas** | 42 |
| **Linhas após limpeza** | **565** |
| **Tipo de problema** | Regressão supervisionada |
| **Variável target** | `salary_in_usd` |

### Colunas do Dataset

| Coluna | Tipo | Descrição |
|---|---|---|
| `work_year` | Numérico | Ano de referência do salário (2020, 2021, 2022) |
| `experience_level` | Categórico | Nível de experiência: EN (Entry), MI (Mid), SE (Senior), EX (Executive) |
| `employment_type` | Categórico | Tipo de emprego: FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance) |
| `job_title` | Categórico | Cargo/função (~50 categorias distintas) |
| `salary` | Numérico | Salário na moeda original — **REMOVIDO (data leakage)** |
| `salary_currency` | Categórico | Moeda original — **REMOVIDO (data leakage)** |
| `salary_in_usd` | Numérico | Salário convertido em USD — **TARGET** |
| `employee_residence` | Categórico | País de residência do funcionário |
| `remote_ratio` | Numérico | Percentual de trabalho remoto (0, 50 ou 100) |
| `company_location` | Categórico | País da empresa |
| `company_size` | Categórico | Tamanho da empresa: S (Small), M (Medium), L (Large) |

---

## 2. Pipeline de Pré-processamento Implementado

O pipeline segue a sequência:

```
Dados Brutos
    ↓
1. Limpeza (Removçao índice, data leakage, nulos, duplicatas)
    ↓
2. Feature Engineering (job_group, is_us_company)
    ↓
3. Separacão de Features (Previsores vs Target)
    ↓
4. Encoding de Variáveis Categóricas
    ↓
5. Divisão Treino/Teste (75%/25%)
    ↓
6. Padronizacão (opcional)
    ↓
Dados Processados (salvo em CSV)
```

---

## 3. Etapas Detalhadas

### 3.1 — Limpeza dos Dados

#### 3.1.1 Remoção de Coluna de Índice Duplicado

A coluna `Unnamed: 0` aparece quando o CSV é salvo com índice e depois relido. Foi removida condicionalmente:

```python
if 'Unnamed: 0' in base.columns:
    base = base.drop(columns=['Unnamed: 0'])
```

**Justificativa:** É apenas um índice sequencial sem valor preditivo.

---

#### 3.1.2 Remoção de Colunas com Data Leakage

As colunas `salary` (valor na moeda original) e `salary_currency` (código da moeda) foram removidas:

```python
base = base.drop(columns=['salary', 'salary_currency'])
```

**Justificativa:** `salary` é diretamente relacionado ao target `salary_in_usd` (é a mesma informação em moeda diferente). Manter essas colunas causaria **data leakage** — o modelo "veria" a resposta durante o treino, resultando em performance artificialmente alta que não se repete em dados novos.

---

#### 3.1.3 Verificação de Valores Nulos

```python
nulos = pd.isnull(base).any()
```

**Resultado:** Nenhum valor nulo foi encontrado no dataset.

---

#### 3.1.4 Remoção de Linhas Duplicadas

```python
num_duplicated = len(base[base.duplicated()])
base = base.drop_duplicates()
```

**Resultado:** O dataset original tinha **607 linhas** (com coluna de índice extra). Após remoção da coluna índice, o shape passa a 607 × 11. Foram detectadas **42 linhas duplicadas**, resultando em **565 linhas únicas** — alinhado com o notebook de referência do Kaggle.

> ⚠️ **Observação:** Parte das duplicatas é esperada pois o dataset de salários do Kaggle agrega registros de múltiplas fontes, e muitas empresas reportaram combinacões idênticas de cargo/salário/país.

---

### 3.2 — Feature Engineering

Duas novas features foram criadas para capturar informações de alta importância:

#### 3.2.1 Agrupamento de `job_title` em `job_group`

Com ~50 cargos distintos e 565 amostras, muitos cargos têm apenas 1-3 ocorrências. Agrupar em 6 categorias amplas reduz ruído:

| Grupo | Exemplos |
|---|---|
| **Data Scientist** | Data Scientist, Principal Data Scientist, Applied Data Scientist |
| **Data Engineer** | Data Engineer, Big Data Engineer, Cloud Data Engineer, ETL Developer |
| **ML Engineer** | Machine Learning Engineer, NLP Engineer, Computer Vision Engineer |
| **Data Analyst** | Data Analyst, BI Data Analyst, Financial Data Analyst |
| **Manager/Director** | Data Science Manager, Head of Data, Director of Data Engineering |
| **Research/AI** | Research Scientist, AI Scientist |

#### 3.2.2 Feature `is_us_company`

Feature binária: empresa sediada nos EUA (1) ou não (0).

```python
base['is_us_company'] = (base['company_location'] == 'US').astype(int)
```

**Resultado:** 318 de 565 empresas (56,3%) são americanas. A mediana de salário EUA é ~2x maior que outros países ($133K vs $63K).

---

### 3.3 — Separacão de Features (Previsores e Objetivo)

```python
cols_previsores = [
    'work_year',          # Numérico
    'experience_level',   # Categórico → OrdinalEncoder
    'employment_type',    # Categórico → encoding
    'job_group',          # Categórico (agrupado de job_title) → encoding
    'employee_residence', # Categórico → encoding
    'remote_ratio',       # Numérico (0, 50, 100)
    'company_location',   # Categórico → encoding
    'company_size',       # Categórico → encoding
    'is_us_company'       # Binária (nova feature)
]

cols_objetivo = ['salary_in_usd']
```

---

### 3.4 — Encoding de Variáveis Categóricas

Duas estratégias combinadas:

#### 3.4.1 OrdinalEncoder para `experience_level`

```python
oe = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
previsores['experience_level'] = oe.fit_transform(previsores[['experience_level']]).astype(int)
# Resultado: EN=0, MI=1, SE=2, EX=3
```

**Justificativa:** `experience_level` tem ordem natural (Entry < Mid < Senior < Executive). O OrdinalEncoder com a ordem explícita preserva essa relação corretamente.

#### 3.4.2 One-Hot Encoding para nominais (Variação `com_dummy`)

```python
previsores = pd.get_dummies(previsores, columns=colunas_categoricas_nominais)
```

**Resultado:** De 9 features base, o One-Hot Encoding expande para **125 features** (mais compacto que antes pois `job_group` tem 6 categorias vs 50+ do `job_title`).

**Vantagens:**
- Não implica ordem entre categorias nominais
- Funciona bem para modelos lineares

#### 3.4.3 LabelEncoder para nominais (Variação `sem_dummy`)

```python
le = LabelEncoder()
previsores[coluna] = le.fit_transform(previsores[coluna])
```

**Resultado:** Mantém 9 features, cada categórica nominal substituida por inteiros.

**Desvantagem:** Impõe uma ordem arbitrária. Adequado principalmente para modelos baseados em árvores.

---

### 3.5 — Divisão Treino/Teste

```python
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(
    previsores, objetivo,
    test_size=0.25,
    random_state=0
)
```

| Split | Proporção | Observações |
|---|---|---|
| **Treinamento** | 75% | ~184 amostras |
| **Teste** | 25% | ~61 amostras |

**Justificativa do `test_size=0.25`:** Com ~245 linhas disponíveis após limpeza, 75/25 é um split equilibrado que preserva dados suficientes para treino sem reduzir o conjunto de teste a uma amostra muito pequena.

**`random_state=0`:** Garante reprodutibilidade — o mesmo split é gerado em toda execução.

> ⚠️ **Limitação identificada:** Um único split pode ser sensível à distribuição aleatória dos dados. Com apenas ~245 amostras, isso pode levar a estimativas de performance instáveis. **K-Fold Cross Validation** seria mais robusto.

---

### 3.6 — Padronização (Normalização)

Duas variações foram comparadas:

#### 3.5.1 Com StandardScaler (Variação `com_std`)

```python
scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)  # fit + transform no treino
previsores_teste = scaler.transform(previsores_teste)                  # apenas transform no teste
```

**O que faz:** Transforma cada feature para ter **média 0** e **desvio padrão 1**.

**Boa prática aplicada:** O `scaler` é fitado **apenas no conjunto de treinamento** e depois aplicado ao teste. Isso evita data leakage — o modelo não "vê" estatísticas do conjunto de teste durante o treinamento.

**Quando é necessário:**
- Regressão Linear: **Sim** (os coeficientes precisam de escala comparável)
- Regressão Polinomial: **Sim** (especialmente em graus altos)
- Random Forest: **Não necessário** (árvores são invariantes à escala)

#### 3.5.2 Sem Padronização (Variação `sem_std`)

Os dados são usados na escala original. Adequado principalmente para árvores de decisão e Random Forest.

---

### 3.7 — Estrutura das Variações Geradas

O script `pre_processamento_unificado.py` gera as **4 variações**:

| Variação | Encoding | Padronização | Features |
|---|---|---|---|
| `com_dummy_com_std` | OrdinalEncoder + One-Hot | StandardScaler | **125** |
| `com_dummy_sem_std` | OrdinalEncoder + One-Hot | Nenhuma | **125** |
| `sem_dummy_com_std` | OrdinalEncoder + LabelEncoder | StandardScaler | **9** |
| `sem_dummy_sem_std` | OrdinalEncoder + LabelEncoder | Nenhuma | **9** |

Cada variação é salva em sua própria subpasta dentro de `dados_processados/`:

```
dados_processados/
├── com_dummy_com_std/
│   ├── previsores_treinamento.csv
│   ├── previsores_teste.csv
│   ├── objetivo_treinamento.csv
│   ├── objetivo_teste.csv
│   └── config.json
├── com_dummy_sem_std/
│   └── ...
├── sem_dummy_com_std/
│   └── ...
└── sem_dummy_sem_std/
    └── ...
```

---

## 4. Resultados dos Modelos

### 4.1 Regressão Linear Múltipla

| Pré-processamento | Encoding | R² (teste) | MAPE | MAE | RMSE |
|---|---|---|---|---|---|
| `com_dummy_sem_std` | One-Hot | **0.455** | 72.4% | R$ 39.763 | R$ 56.109 |
| `com_dummy_com_std` | One-Hot | 0.453 | 72.3% | R$ 39.709 | R$ 56.190 |
| `sem_dummy_com_std` | OrdinalEncoder+Label | 0.427 | 84.2% | R$ 39.863 | R$ 57.506 |
| `sem_dummy_sem_std` | OrdinalEncoder+Label | 0.427 | 84.2% | R$ 39.863 | R$ 57.506 |

**Observacões:**
- One-Hot Encoding supera Label Encoding em ~3 pontos de R²
- O StandardScaler tem impacto mínimo na Regressão Linear para este dataset
- O MAE nas variaões com FE caiu de ~$40K (versão anterior) para ~$32K

---

### 4.2 Regressão Polinomial (graus 2 e 3)

*(Resultados completos em `regressao_polinomial/resultados_regressao_polinomial.csv`)*

**Configuracão:** Graus testados: **2 e 3**

**Observacões:**
- A Regressão Polinomial com One-Hot Encoding colapsa rapidamente — a explosão de dimensionalidade (125 features × grau polinomial) cria volume inviável
- Com LabelEncoder (9 features), o grau 2 pode apresentar resultado razoável
- Grau 3 com grande número de features resulta em overfitting severo (R² negativo no teste)
- **Conclusão:** A Regressão Polinomial não é adequada para este dataset com One-Hot Encoding

---

### 4.3 Random Forest

*(Script implementado — resultados gerados em `random_forest/resultados_random_forest.csv`)*

O Random Forest foi configurado com grid:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15, 20]

---

## 5. Fluxo dos Scripts de Modelos

Todos os modelos (Regressão Linear, Polinomial, Árvore de Decisão, Random Forest) seguem o mesmo fluxo:

```
Etapa 1: Rodar modelo em TODAS as bases -> Preencher CSV de resultados
    ↓
Etapa 2: Analisar CSV -> Encontrar melhor R² -> Gerar plot SOMENTE do melhor
    ↓
Etapa 3: Adicionar melhor resultado ao arquivo consolidado (melhores_resultados.csv)
```

**Colunas padrao dos CSVs de resultado:**

| Coluna | Descrição |
|---|---|
| `preprocessamento` | Nome da variação de pré-processamento |
| `encoding` | Tipo de encoding (`one_hot` ou `label_encoder`) |
| `padronizacao` | Tipo de padronização (`StandardScaler` ou `nenhuma`) |
| *(config específica do modelo)* | Ex: `grau`, `max_depth`, `n_estimators` |
| `r2_score` | R² no conjunto de teste |
| `mape` | Mean Absolute Percentage Error (%) |
| `mae` | Mean Absolute Error (USD) |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error (USD) |

---

## 6. O que Ainda Pode ser Melhorado

### 6.1 Validação Cruzada (K-Fold)
**Status:** Nao implementado nos scripts principais 
Com 565 amostras, um único split 75/25 pode ser instável. K-Fold com k=5 melhoraria a confiabilidade das estimativas.

### 6.2 MinMaxScaler como Alternativa
**Status:** Nao implementado 
Alternativa ao StandardScaler que escala dados para [0, 1].

### 6.3 Feature Selection
**Status:** Nao implementado 
Após One-Hot Encoding, há 125 colunas. RFE ou análise de importância poderiam reduzir a dimensionalidade.

---

## 7. Referências e Scripts

| Script | Localização | Descrição |
|---|---|---|
| `pre_processamento_unificado.py` | `pre_processamento/` | Gera as 4 variações de pré-processamento |
| `pre_processamento_com_dummy_com_std.py` | `pre_processamento/` | Variação individual: One-Hot + StandardScaler |
| `pre_processamento_com_dummy_sem_std.py` | `pre_processamento/` | Variação individual: One-Hot, sem padronização |
| `pre_processamento_sem_dummy_com_std.py` | `pre_processamento/` | Variação individual: LabelEncoder + StandardScaler |
| `pre_processamento_sem_dummy_sem_std.py` | `pre_processamento/` | Variação individual: LabelEncoder, sem padronização |
| `regressao_linear.py` | `regressao_linear/` | Regressão Linear Múltipla em todas as variações |
| `regressao_polinomial.py` | `regressao_polinomial/` | Regressão Polinomial, graus 2-7 |
| `random_forest.py` | `random_forest/` | Random Forest com grid de hiperparâmetros |
| `executar_tudo.py` | raiz do projeto | Executa todos os scripts em sequência |

---

*Documentação gerada em 26/05/2026 — Projeto de Machine Learning | Jogos Digitais - UFMS*
