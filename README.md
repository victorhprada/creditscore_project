# Projeto de Credit Score (Pontuação de Crédito)

## Visão Geral

Este projeto realiza a **análise de Credit Score** usando técnicas de machine learning para classificar clientes em diferentes categorias de risco de crédito. O objetivo é prever a classificação do score de crédito de um cliente (**Baixo**, **Médio** ou **Alto**) com base em características demográficas e financeiras, auxiliando instituições financeiras a tomar decisões informadas sobre aprovação de crédito e avaliação de risco.

---

## Contexto de Negócio

O **Credit Score** (pontuação de crédito) é uma representação numérica da capacidade de um indivíduo de cumprir suas obrigações financeiras, como pagamentos de empréstimos, faturas de cartão de crédito, entre outros. Este modelo ajuda:

- **Instituições financeiras** a avaliar o risco de emprestar dinheiro para um determinado indivíduo
- **Bancos e credores** a tomar decisões informadas sobre aprovação ou negação de crédito
- **Emprestadores** a definir termos e condições adequados para empréstimos

### Regras de Negócio & Insights

Com base na análise exploratória dos dados, foram identificados os seguintes padrões de negócio:

1. **Distribuição por Gênero**: A maioria dos clientes são mulheres, o que pode indicar uma maior taxa de análise de crédito entre mulheres
2. **Nível de Escolaridade**: A maioria dos clientes possui graduação (Bacharelado), sugerindo que maior escolaridade correlaciona-se com maior engajamento de crédito
3. **Estado Civil**: A maioria dos clientes é casada, potencialmente indicando perfis financeiros mais estáveis
4. **Propriedade de Imóvel**: A maioria dos clientes possui casa própria, um indicador positivo de capacidade de crédito
5. **Distribuição do Score**: A maioria dos clientes possui score de crédito alto, indicando uma base de clientes geralmente saudável

---

## Dataset

### Fonte
`CREDIT_SCORE_PROJETO_PARTE1.csv` — Dados demográficos e financeiros de clientes

### Variáveis (Features)

| Variável | Descrição | Tipo |
|----------|-----------|------|
| **Age** | Idade do cliente | Numérica |
| **Gender** | Gênero do cliente (Masculino/Feminino) | Categórica |
| **Income** | Salário mensal (formato brasileiro) | Numérica |
| **Education** | Nível de escolaridade (Ensino Médio, Associado, Bacharelado, Mestrado, Doutorado) | Categórica |
| **Marital Status** | Estado civil (Solteiro/Casado) | Categórica |
| **Number of Children** | Quantidade de filhos | Numérica |
| **Home Ownership** | Tipo de moradia (Própria/Alugada) | Categórica |
| **Credit Score** | Variável alvo (Baixo/Médio/Alto) | Categórica |

---

## Como o Código Funciona

### 1. Carregamento dos Dados e Inspeção Inicial

```python
df = pd.read_csv('CREDIT_SCORE_PROJETO_PARTE1.csv', delimiter=';')
```

- Carrega o dataset e exibe as primeiras linhas
- Verifica os tipos de dados de todas as colunas

### 2. Pré-Processamento dos Dados

#### Transformação da Coluna Income
A coluna `Income` vem no formato brasileiro (ex: `50.000,00`). O código:
- Remove os separadores de milhares (`.`)
- Remove os separadores decimais (`,`)
- Converte para o tipo `float`

#### Tratamento de Valores Faltantes
- Identifica valores nulos em todas as colunas
- Apenas a coluna `Age` (Idade) possui valores faltantes
- **Estratégia**: Calcula a assimetria (skewness) da distribuição de `Age`
  - Se assimetria ≈ 0 (distribuição quase normal) → Preenche com a **média**
  - Essa abordagem preserva o formato da distribuição sem introduzir viés

#### Validação de Dados Categóricos
Verifica se há valores incorretamente inseridos nas colunas categóricas:
- `Gender`, `Education`, `Marital Status`, `Home Ownership`, `Credit Score`

### 3. Análise Exploratória dos Dados (EDA)

#### Análise Univariada
- **Resumo Estatístico**: Usa `describe()` para entender as distribuições das variáveis numéricas
- **Detecção de Outliers**: Box plots para `Age` (Idade), `Income` (Renda) e `Number of Children` (Número de Filhos)
- **Distribuições Categóricas**: Gráficos de barras com porcentagens para:
  - Distribuição por gênero
  - Níveis de escolaridade
  - Estado civil
  - Propriedade de imóvel
  - Categorias de credit score

#### Análise Bivariada
Explora relacionamentos entre variáveis:
- **Idade vs Estado Civil**: Mediana de idade por estado civil
- **Credit Score vs Escolaridade**: Heatmap mostrando correlação entre nível de escolaridade e score de crédito
- **Renda vs Idade**: Tendência de renda mediana por faixa etária
- **Renda vs Credit Score**: Renda média por categoria de score de crédito
- **Propriedade de Imóvel vs Credit Score**: Heatmap analisando o impacto da posse de imóvel
- **Escolaridade vs Estado Civil**: Gráfico de barras empilhadas mostrando relação entre educação e casamento
- **Gênero vs Propriedade de Imóvel**: Heatmap analisando padrões de gênero e propriedade
- **Escolaridade vs Renda**: Renda média por nível de escolaridade

#### Análise de Correlação
- Gera heatmaps de correlação antes e depois da codificação
- Identifica multicolinearidade e relacionamentos entre features

### 4. Engenharia de Features

#### Label Encoding
Aplicado em variáveis categóricas ordinais/binárias:
- `Gender` → `Gender_encoded`
- `Marital Status` → `Marital_Status_encoded`
- `Home Ownership` → `Home_Ownership_encoded`

#### One-Hot Encoding
Aplicado em `Education` com `drop_first=True` para evitar a armadilha da variável dummy:
- Cria colunas binárias para cada nível de escolaridade

### 5. Separação Treino-Teste

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
```

- **75%** dos dados para treinamento
- **25%** dos dados para teste
- `random_state` fixo para reprodutibilidade

### 6. Tratamento do Desequilíbrio de Classes (SMOTE)

O dataset provavelmente possui classes desequilibradas de credit score. O código aplica **SMOTE (Synthetic Minority Over-sampling Technique)**:

- Converte features categóricas para numéricas usando `get_dummies()`
- Gera amostras sintéticas para classes minoritárias
- Balanceia o dataset de treinamento para melhorar o desempenho do modelo em todas as classes

---

## Dependências

```
pandas
seaborn
matplotlib
plotly
scipy
scikit-learn
imbalanced-learn (imblearn)
```

### Instalação

```bash
pip install pandas seaborn matplotlib plotly scipy scikit-learn imbalanced-learn
```

---

## Como Executar

1. Certifique-se de que o arquivo de dados `CREDIT_SCORE_PROJETO_PARTE1.csv` está na raiz do projeto
2. Execute o script:

```bash
python credit_score.py
```

Ou use o Jupyter Notebook para exploração interativa:

```bash
jupyter notebook "Profissao Cientista de Dados M17 Projeto.ipynb"
```

---

## Estrutura do Projeto

```
creditscore_project/
├── CREDIT_SCORE_PROJETO_PARTE1.csv  # Dataset bruto
├── credit_score.py                  # Script principal de análise
├── Profissao Cientista de Dados M17 Projeto.ipynb  # Jupyter Notebook
└── README.md                        # Documentação do projeto
```

---

## Próximos Passos

Este projeto cobre **pré-processamento de dados e análise exploratória** (Parte 1). Melhorias futuras incluem:

- [ ] Treinamento de modelos (Random Forest, XGBoost, Regressão Logística)
- [ ] Ajuste de hiperparâmetros
- [ ] Métricas de avaliação do modelo (Acurácia, Precisão, Recall, F1-Score, ROC-AUC)
- [ ] Análise de importância das features
- [ ] Pipeline de deploy do modelo
- [ ] Validação cruzada para avaliação robusta de desempenho

---

## Autor

Desenvolvido como parte do **Curso Profissional de Cientista de Dados** - Projeto Módulo 17

---

## Licença

Este projeto é para fins educacionais.
