# -*- coding: utf-8 -*-
"""
Pré-processamento Unificado
Dataset: ds_salaries.csv
Target: salary_in_usd

Executa TODAS as 4 variações de pré-processamento de uma só vez:
  1. com_dummy_com_std   (One-Hot + StandardScaler)
  2. com_dummy_sem_std   (One-Hot, sem padronização)
  3. sem_dummy_com_std   (LabelEncoder + StandardScaler)
  4. sem_dummy_sem_std   (LabelEncoder, sem padronização)

Cada variação é salva em sua própria subpasta dentro de dados_processados/.
"""

import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

################## Configuração de diretórios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

caminho_csv = os.path.join(_dir, '..', 'ds_salaries.csv')
dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')

################## Leitura e limpeza dos dados ##################

base = pd.read_csv(caminho_csv)
print(f'Shape original: {base.shape}')

# Removendo linhas duplicadas
num_duplicated = len(base[base.duplicated()])
print(f'STATUS: There are {num_duplicated} duplicated rows')
base = base.drop_duplicates()
print(f'After removing duplicates: {base.shape}')

# Removendo colunas que causam data leakage (salary e salary_currency)
base = base.drop(columns=['salary', 'salary_currency'])

# Removendo coluna de índice se existir
if 'Unnamed: 0' in base.columns:
    base = base.drop(columns=['Unnamed: 0'])

# Procurando as colunas que possuem algum valor faltante
nulos = pd.isnull(base).any()
if nulos.any():
    print(f'Colunas com valores nulos: {nulos[nulos].index.tolist()}')
else:
    print('Nenhum valor nulo encontrado.')

################## Separando dados em previsores e objetivo ##################

cols_previsores = ['work_year', 'experience_level', 'employment_type',
                   'job_title', 'employee_residence', 'remote_ratio',
                   'company_location', 'company_size']

cols_objetivo = ['salary_in_usd']

colunas_categoricas = ['experience_level', 'employment_type', 'job_title',
                       'employee_residence', 'company_location', 'company_size']

################## Definição das variações ##################

VARIACOES = [
    {'nome': 'com_dummy_com_std',  'dummy': True,  'std': True},
    {'nome': 'com_dummy_sem_std',  'dummy': True,  'std': False},
    {'nome': 'sem_dummy_com_std',  'dummy': False, 'std': True},
    {'nome': 'sem_dummy_sem_std',  'dummy': False, 'std': False},
]

################## Processamento de cada variação ##################

for var in VARIACOES:
    nome = var['nome']
    usar_dummy = var['dummy']
    usar_std = var['std']

    print(f'\n{"="*60}')
    print(f'  Processando: {nome}')
    print(f'{"="*60}')

    previsores = base[cols_previsores].copy()
    objetivo = base[cols_objetivo].copy()

    # Encoding
    if usar_dummy:
        previsores = pd.get_dummies(previsores, columns=colunas_categoricas)
        print(f'  Encoding: One-Hot (dummy) -> {previsores.shape[1]} features')
    else:
        label_encoders = {}
        for coluna in colunas_categoricas:
            le = LabelEncoder()
            previsores[coluna] = le.fit_transform(previsores[coluna])
            label_encoders[coluna] = le
        print(f'  Encoding: LabelEncoder -> {previsores.shape[1]} features')

    # Train/test split
    previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(
        previsores, objetivo, test_size=0.25, random_state=0
    )

    # Padronização
    if usar_std:
        scaler = StandardScaler()
        previsores_treinamento = scaler.fit_transform(previsores_treinamento)
        previsores_teste = scaler.transform(previsores_teste)
        print('  Padronizacao: StandardScaler aplicado')
    else:
        print('  Padronizacao: Nenhuma')

    # Salvando em subpasta própria
    dados_dir = os.path.join(dados_dir_raiz, nome)
    os.makedirs(dados_dir, exist_ok=True)

    pd.DataFrame(previsores_treinamento).to_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'), index=False)
    pd.DataFrame(previsores_teste).to_csv(os.path.join(dados_dir, 'previsores_teste.csv'), index=False)
    objetivo_treinamento.to_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'), index=False)
    objetivo_teste.to_csv(os.path.join(dados_dir, 'objetivo_teste.csv'), index=False)

    config = {'NOME_PREPROCESSAMENTO': nome}
    with open(os.path.join(dados_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    print(f'  [OK] Salvo em: {os.path.abspath(dados_dir)}')

################## Resumo final ##################

print(f'\n{"="*60}')
print('[OK] Todas as 4 variacoes processadas com sucesso!')
print(f'Dados salvos em: {os.path.abspath(dados_dir_raiz)}')
print(f'{"="*60}')
