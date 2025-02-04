import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('CREDIT_SCORE_PROJETO_PARTE1.csv', delimiter=';')

print(df.head(10))

# Verificando os tipos de dados
print(df.dtypes)

# A coluna Income será alterada para valores do tipo float
df['Income'] = df['Income'].str.replace('.', '', regex=False) # Removendo pontos (.) 
df['Income'] = df['Income'].str.replace(',', '', regex=False) # Removendo virgulas (,)
df['Income'] = df['Income'].astype(float)
print(df.dtypes)

# Verifique se temos colunas com dados faltantes. 
# Caso existam colunas com dados faltantes faça o tratamento desses dados, excluindo ou substituindo esses valores. 
# Justifique sua escolha

print(df.isnull().sum()) # Trazendo valores nulos por coluna
print((df['Age'].isnull().sum() / len(df['Age']) * 100)) # Trazendo a porcentagem dos valores nulos na coluda de Idade (Apenas ela possuí valores nulos)

# Verificando assimetria da idade, uma assimetria leve é indicado preencher os dados nulos com a média
assimetria_age = skew(df['Age'].dropna())
print(f"Assimetria: {assimetria_age:.2f}")

# Assimetria perto de 0 (zero) quer dizer que a distribuição é quase normal, então podemos preencher os valores nulos com a média
media_age = df['Age'].mean()

df.fillna({'Age': media_age}, inplace=True)
print(df['Age'].isnull().sum())

# Verifique se temos valores digitados de forma incorreta nas variáveis categóricas que necessitem de tratamento
print(df[['Gender', 'Education', 'Marital Status', 'Home Ownership', 'Credit Score']].apply(pd.Series.unique))

# Inicio da análise univariada
print(df.describe())

# Buscando possíveis outliers com o Box Plot
df.boxplot(column='Age')
plt.title('Box Plot Idade')
plt.ylabel('Valores')
plt.show()

df.boxplot(column='Income')
plt.title('Box Plot Income')
plt.ylabel('Valores')
plt.show()

# Outlier encontrado no número de filhos 
df.boxplot(column='Number of Children')
plt.title('Box Plot Filhos')
plt.ylabel('Valores')
plt.show()

# Número de filhos
count_children = df['Number of Children'].value_counts() # Contagem dos valores da coluna número de filhos


# Calculando a porcentagem dos números de filhos
porcent_children = (count_children / count_children.sum()) * 100
ax = porcent_children.plot(kind='bar')
plt.title('Gráfico de barras para a variável número de filhos')
plt.xlabel('Número de filhos')
plt.ylabel('Frequência')
plt.show()

# Analisando as variáveis categóricas

# Genêro
# Os clientes em sua maioria são mulheres, isso pode apresentar uma taxa de análise de crédito maior nas mulheres do que nos homens.
# Calculando a porcentagem 
count_gender = df['Gender'].value_counts()
percent_gender = (count_gender / count_gender.sum()) * 100

ax = count_gender.plot(kind='bar')
# Adicionando porcentagem as barras
for i, v in enumerate(count_gender):
    ax.text(i, v + 1, f'{percent_gender[i]:.2f}%', ha='center')
plt.title('Gráfico de barras para a variável genêro')
plt.xlabel('Genereo')
plt.ylabel('Frequência')
plt.show()

# Educação
# Os clientes em sua maioria possuêm grau de bacharelado, isso pode apresentar uma taxa de análise de crédito maior nas pessoas com nível de bacharelado.
count_edu = df['Education'].value_counts()
percent_edu = (count_edu / count_edu.sum()) * 100

ax = count_edu.plot(kind='bar')
# Adicionando porcentagem as barras
for i, v in enumerate(count_edu):
    ax.text(i, v + 1, f'{percent_edu[i]:.2f}%', ha='center')
plt.title('Gráfico de barras para a variável educação')
plt.xlabel('Grau de Educação')
plt.ylabel('Frequência')
plt.show()

# Status matrimonial
# Os clientes em sua maioria são casados, isso pode apresentar uma taxa de análise de crédito maior nas pessoas casadas.

count_matrial_status = df['Marital Status'].value_counts()
percent_matrial_status = (count_matrial_status / count_matrial_status.sum()) * 100

ax = count_matrial_status.plot(kind='bar')
for i, v in enumerate(count_matrial_status):
    ax.text(i, v + 1, f'{percent_matrial_status[i]:.2f}%', ha='center')
plt.title('Gráfico de barras para o status matrimonial')
plt.xlabel('Status Matrimonial')
plt.ylabel('Frequência')
plt.show()

# Home Ownership
# Os clientes em sua maioria possuem casa prórpia, isso pode apresentar uma taxa de análise de crédito maior nas pessoas que já possuem uma casa própria.

count_ownership = df['Home Ownership'].value_counts()
percent_ownership = (count_ownership / count_ownership.sum()) * 100

ax = count_ownership.plot(kind='bar')
for i, v in enumerate(count_ownership):
    ax.text(i, v + 1, f'{percent_ownership[i]:.2f}%', ha='center')
plt.title('Gráfico de barras para o casa própria')
plt.xlabel('Casa Própria')
plt.ylabel('Frequência')
plt.show()

# Credit Score
# Os clientes em sua maioria possuem um score alto.

count_credit_score = df['Credit Score'].value_counts()
percent_credit_score = (count_credit_score / count_credit_score.sum()) * 100

ax = count_credit_score.plot(kind='bar')
for i, v in enumerate(count_credit_score):
    ax.text(i, v + 1, f'{percent_credit_score[i]:.2f}%', ha='center')
plt.title('Gráfico de barras para score de crédito')
plt.xlabel('Score de Crédito')
plt.ylabel('Frequência')
plt.show()

# Bivariada

# Idade x Estado Civil
mediana_idade_estado_civil = df.groupby('Marital Status')['Age'].median().reset_index()

fig = px.bar(mediana_idade_estado_civil,
             x='Marital Status',
             y='Age',
             title='Media de idade por estado civil',
             labels={'Marital Status': 'Estado Civil', 'Age': 'Idade'})
fig.show()

# Score de Crédito e Nível de Escolaridade

# Aplicando o Label Encoder na coluna de crédito score
label_enconder = LabelEncoder()

df['Credit_Score_encoded'] = label_enconder.fit_transform(df['Credit Score'])

# Mapeando manualmente a ordem
mapping = {
    0: 'Average',
    1: 'High',
    2: 'Low'
}

# Criando uma nova coluna com os rótulos desejados
df['Credit_Score_label'] = df['Credit_Score_encoded'].map(mapping)

# Adicionando ordem categorica
df['Credit_Score_label'] = pd.Categorical(df['Credit_Score_label'],
                                            categories=['High', 'Average', 'Low'],
                                            ordered=True)

# Criando tabela de contingência usando a coluna com os rótulos
table = pd.crosstab(df['Credit_Score_label'], df['Education'])

plt.figure(figsize=(10, 6))
sn.heatmap(table, annot=True, fmt='d', cmap='coolwarm')
plt.title('Mapa de Calor: Score de Crédito x Nível de Escolaridade')
plt.xlabel('Education')
plt.ylabel('Credit Score')
plt.show()

print(label_enconder.classes_)

# Salário x Idade
mediana_salario_idade = df.groupby('Age')['Income'].median().reset_index()

fig = px.line(mediana_salario_idade,
             x='Age',
             y='Income',
             title='Media de salário por idade',
             labels={'Age': 'Idade', 'Income': 'Média de Salário'})
fig.show()

# Salário x Score
mediana_salario_score = df.groupby('Credit_Score_encoded')['Income'].mean().reset_index()

fig = px.bar(mediana_salario_score,
             x='Credit_Score_encoded',
             y='Income',
             title='Média de salários x Score de Crédito')
fig.show()

# Casa prórpia x Credit Score

# Criando tabela de contingência
table_casa_credito = pd.crosstab(df['Credit Score'], df['Home Ownership'])

plt.figure(figsize=(10, 6))
sn.heatmap(table_casa_credito, annot=True, fmt='d', cmap='coolwarm')
plt.title('Mapa de Calor: Credit Score x Casa Própria')
plt.xlabel('Casa Própria')
plt.ylabel('Credit Score')
plt.show()