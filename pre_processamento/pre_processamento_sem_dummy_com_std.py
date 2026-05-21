# -*- coding: utf-8 -*-
"""
Pré-processamento - Sem Dummy, Com Padronização
Dataset: ds_salaries.csv
Target: salary_in_usd

Encoding: LabelEncoder
Padronização: StandardScaler
"""

import pandas as pd

################## Pré-processamento dos dados ##################

NOME_PREPROCESSAMENTO = 'sem_dummy_com_std'

# Leitura dos dados
base = pd.read_csv('../ds_salaries.csv')
base.describe()

# =============================================================================
#                     Tratando valores inválidos
# =============================================================================

# Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()

# Removendo colunas que causam data leakage (salary e salary_currency)
base = base.drop(columns=['salary', 'salary_currency'])

# Removendo coluna de índice se existir
if 'Unnamed: 0' in base.columns:
    base = base.drop(columns=['Unnamed: 0'])

# Removendo linhas duplicadas
num_duplicated = len(base[base.duplicated()])
print(f'STATUS: There are {num_duplicated} duplicated rows')
print(f'Shape antes: {base.shape}')
base = base.drop_duplicates()
print(f'After removing duplicates: {base.shape}')

# =============================================================================
#                     Separando dados em previsores e objetivo
# =============================================================================

cols_previsores = ['work_year', 'experience_level', 'employment_type',
                   'job_title', 'employee_residence', 'remote_ratio',
                   'company_location', 'company_size']

cols_objetivo = ['salary_in_usd']
previsores = base[cols_previsores].copy()
objetivo = base[cols_objetivo]

# =============================================================================
#      Transformar as variáveis categóricas em valores numéricos
# =============================================================================

# Usando LabelEncoder para cada coluna categórica
from sklearn.preprocessing import LabelEncoder

colunas_categoricas = ['experience_level', 'employment_type', 'job_title',
                       'employee_residence', 'company_location', 'company_size']

label_encoders = {}
for coluna in colunas_categoricas:
    le = LabelEncoder()
    previsores[coluna] = le.fit_transform(previsores[coluna])
    label_encoders[coluna] = le

# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================

# Usando 25% para teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores,
                                                                                                  objetivo,
                                                                                                  test_size=0.25,
                                                                                                  random_state=0)

# =============================================================================
#                     Padronização dos dados
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

# =============================================================================
#          Salvando dados processados para uso nos scripts de regressão
# =============================================================================

import os
import json

try:
    _dir_pre = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir_pre = os.getcwd()

# Pasta compartilhada na raiz do projeto
dados_dir = os.path.join(_dir_pre, '..', 'dados_processados')
os.makedirs(dados_dir, exist_ok=True)

# Salvando os datasets de treino e teste
pd.DataFrame(previsores_treinamento).to_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'), index=False)
pd.DataFrame(previsores_teste).to_csv(os.path.join(dados_dir, 'previsores_teste.csv'), index=False)
objetivo_treinamento.to_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'), index=False)
objetivo_teste.to_csv(os.path.join(dados_dir, 'objetivo_teste.csv'), index=False)

# Salvando configuração do pré-processamento atual
config = {'NOME_PREPROCESSAMENTO': NOME_PREPROCESSAMENTO}
with open(os.path.join(dados_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

print(f'\n✅ Dados do pré-processamento "{NOME_PREPROCESSAMENTO}" salvos em: {os.path.abspath(dados_dir)}')
print('Agora você pode executar diretamente os scripts de regressão linear e polinomial.')
