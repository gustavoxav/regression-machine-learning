# -*- coding: utf-8 -*-
"""
Pre-processamento Unificado
Dataset: ds_salaries.csv
Target: salary_in_usd

Gera as 4 variacoes padrao de pre-processamento:
  1. com_dummy_com_std   (One-Hot + StandardScaler)
  2. com_dummy_sem_std   (One-Hot, sem padronizacao)
  3. sem_dummy_com_std   (LabelEncoder + StandardScaler)
  4. sem_dummy_sem_std   (LabelEncoder, sem padronizacao)

Melhorias aplicadas:
  - OrdinalEncoder para experience_level (EN=0, MI=1, SE=2, EX=3)
  - Agrupamento de job_title em 6 grupos (job_group)
  - Agrupamento de paises em regioes geograficas (employee_region, company_region)
  - Feature binaria is_us_company
  - Sem remocao de outliers
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

################## Configuracao de diretorios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

caminho_csv = os.path.join(_dir, '..', 'ds_salaries.csv')
dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')

################## Leitura e limpeza dos dados ##################

base = pd.read_csv(caminho_csv)
print(f'Shape original: {base.shape}')

# Removendo coluna de indice se existir
if 'Unnamed: 0' in base.columns:
    base = base.drop(columns=['Unnamed: 0'])

# Removendo colunas que causam data leakage
base = base.drop(columns=['salary', 'salary_currency'])

# Verificando valores nulos
nulos = pd.isnull(base).any()
if nulos.any():
    print(f'Colunas com valores nulos: {nulos[nulos].index.tolist()}')
else:
    print('Nenhum valor nulo encontrado.')

# Removendo linhas duplicadas
num_duplicated = len(base[base.duplicated()])
print(f'STATUS: There are {num_duplicated} duplicated rows')
print(f'STATUS: Dimension of "df" = {base.shape}')
base = base.drop_duplicates()
print(f'STATUS: Dimension of "After removing duplicates" = {base.shape}')

################## Feature Engineering ##################

# --- 1. Agrupamento de job_title em grupos amplos ---
grupos_cargo = {
    'Data Scientist': [
        'Data Scientist', 'Applied Data Scientist', 'Principal Data Scientist',
        'Lead Data Scientist', 'Staff Data Scientist', 'Data Science Consultant',
        'Data Science Engineer', 'Data Specialist'
    ],
    'Data Engineer': [
        'Data Engineer', 'Big Data Engineer', 'Cloud Data Engineer',
        'Lead Data Engineer', 'Principal Data Engineer', 'Data Engineering Manager',
        'Director of Data Engineering', 'Big Data Architect', 'Data Architect',
        'ETL Developer', 'Analytics Engineer', 'Data Analytics Engineer',
        'Data Analytics Lead', 'Machine Learning Infrastructure Engineer'
    ],
    'ML Engineer': [
        'Machine Learning Engineer', 'ML Engineer', 'Lead Machine Learning Engineer',
        'Machine Learning Scientist', 'Machine Learning Manager',
        'Machine Learning Developer', 'Head of Machine Learning',
        'Applied Machine Learning Scientist', 'NLP Engineer',
        'Computer Vision Engineer', 'Computer Vision Software Engineer',
        '3D Computer Vision Researcher'
    ],
    'Data Analyst': [
        'Data Analyst', 'BI Data Analyst', 'Business Data Analyst',
        'Lead Data Analyst', 'Principal Data Analyst', 'Marketing Data Analyst',
        'Financial Data Analyst', 'Finance Data Analyst', 'Data Analytics Manager',
        'Data Analytics Lead'
    ],
    'Manager/Director': [
        'Data Science Manager', 'Director of Data Science', 'Head of Data Science',
        'Head of Data', 'Data Engineering Manager', 'Data Analytics Manager',
        'Director of Data Engineering'
    ],
    'Research/AI': [
        'Research Scientist', 'AI Scientist', 'Applied Machine Learning Scientist'
    ]
}

def mapear_cargo(titulo):
    for grupo, titulos in grupos_cargo.items():
        if titulo in titulos:
            return grupo
    return 'Other'

base['job_group'] = base['job_title'].apply(mapear_cargo)

# --- 2. Feature binaria: empresa sediada nos EUA ---
base['is_us_company'] = (base['company_location'] == 'US').astype(int)

# --- 3. Agrupamento de paises em regioes geograficas ---
# Reduz cardinalidade de ~50 paises para 7 regioes, evitando
# features esparsas (one-hot) ou encodings arbitrarios (label).
regioes = {
    'North_America':     ['US', 'CA'],
    'Latin_America':     ['MX', 'BR', 'AR', 'CL', 'CO', 'BO', 'HN', 'PR'],
    'Western_Europe':    ['GB', 'DE', 'FR', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH',
                          'IE', 'LU', 'DK', 'SE', 'FI', 'NO', 'PT'],
    'Eastern_Europe':    ['PL', 'RO', 'CZ', 'HU', 'HR', 'SI', 'RS', 'UA', 'EE',
                          'LT', 'LV', 'BG', 'MT', 'MD', 'GR'],
    'South_Asia':        ['IN', 'PK', 'LK', 'BD'],
    'East_Asia_Pacific': ['JP', 'CN', 'KR', 'HK', 'SG', 'MY', 'TW', 'VN',
                          'PH', 'TH', 'ID', 'AU', 'NZ'],
    'Middle_East_Africa':['AE', 'IL', 'TR', 'NG', 'KE', 'DZ', 'IQ', 'IR',
                          'SA', 'EG', 'RU'],
}

def mapear_regiao(pais):
    for regiao, paises in regioes.items():
        if pais in paises:
            return regiao
    return 'Other'

base['employee_region'] = base['employee_residence'].apply(mapear_regiao)
base['company_region'] = base['company_location'].apply(mapear_regiao)
print(f'Regioes de residencia: {base["employee_region"].nunique()} categorias')
print(f'Regioes de empresa:    {base["company_region"].nunique()} categorias')

################## Separando features e target ##################

# Usa job_group no lugar de job_title (menor cardinalidade)
cols_previsores = ['work_year', 'experience_level', 'employment_type',
                   'job_group', 'employee_region', 'remote_ratio',
                   'company_region', 'company_size', 'is_us_company']

cols_objetivo = ['salary_in_usd']

colunas_categoricas_nominais = ['employment_type', 'job_group',
                                'employee_region', 'company_region',
                                'company_size']

################## Definicao das variacoes ##################

VARIACOES = [
    {'nome': 'com_dummy_com_std',  'dummy': True,  'std': True},
    {'nome': 'com_dummy_sem_std',  'dummy': True,  'std': False},
    {'nome': 'sem_dummy_com_std',  'dummy': False, 'std': True},
    {'nome': 'sem_dummy_sem_std',  'dummy': False, 'std': False},
]

################## Processamento de cada variacao ##################

for var in VARIACOES:
    nome       = var['nome']
    usar_dummy = var['dummy']
    usar_std   = var['std']

    print(f'\n{"="*60}')
    print(f'  Processando: {nome}')
    print(f'{"="*60}')

    previsores = base[cols_previsores].copy()
    objetivo   = base[cols_objetivo].copy()

    # --- Encoding ---
    # OrdinalEncoder para experience_level (ordem natural: EN < MI < SE < EX)
    oe = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
    previsores['experience_level'] = oe.fit_transform(
        previsores[['experience_level']]
    ).astype(int)

    if usar_dummy:
        previsores = pd.get_dummies(previsores, columns=colunas_categoricas_nominais)
        n_features = previsores.shape[1]
        encoding_label = 'one_hot'
        print(f'  Encoding: OrdinalEncoder(exp_level) + One-Hot -> {n_features} features')
    else:
        for coluna in colunas_categoricas_nominais:
            le = LabelEncoder()
            previsores[coluna] = le.fit_transform(previsores[coluna])
        n_features = previsores.shape[1]
        encoding_label = 'label_encoder'
        print(f'  Encoding: OrdinalEncoder(exp_level) + LabelEncoder -> {n_features} features')

    # --- Train/test split ---
    (previsores_treinamento, previsores_teste,
     objetivo_treinamento, objetivo_teste) = train_test_split(
        previsores, objetivo, test_size=0.25, random_state=0
    )
    print(f'  Split: {len(previsores_treinamento)} treino | {len(previsores_teste)} teste')

    # --- Padronizacao ---
    if usar_std:
        scaler = StandardScaler()
        previsores_treinamento_out = scaler.fit_transform(previsores_treinamento)
        previsores_teste_out       = scaler.transform(previsores_teste)
        std_label = 'StandardScaler'
        print('  Padronizacao: StandardScaler (fit apenas no treino)')
    else:
        previsores_treinamento_out = previsores_treinamento.values
        previsores_teste_out       = previsores_teste.values
        std_label = 'nenhuma'
        print('  Padronizacao: Nenhuma')

    # --- Salvando ---
    col_names = list(previsores.columns)
    dados_dir = os.path.join(dados_dir_raiz, nome)
    os.makedirs(dados_dir, exist_ok=True)

    pd.DataFrame(previsores_treinamento_out, columns=col_names).to_csv(
        os.path.join(dados_dir, 'previsores_treinamento.csv'), index=False)
    pd.DataFrame(previsores_teste_out, columns=col_names).to_csv(
        os.path.join(dados_dir, 'previsores_teste.csv'), index=False)
    objetivo_treinamento.to_csv(
        os.path.join(dados_dir, 'objetivo_treinamento.csv'), index=False)
    objetivo_teste.to_csv(
        os.path.join(dados_dir, 'objetivo_teste.csv'), index=False)

    config = {
        'NOME_PREPROCESSAMENTO': nome,
        'encoding': encoding_label,
        'padronizacao': std_label,
        'n_amostras': int(len(base)),
        'n_treino': int(len(previsores_treinamento)),
        'n_teste': int(len(previsores_teste)),
        'n_features': int(n_features)
    }
    with open(os.path.join(dados_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'  [OK] Salvo em: {os.path.abspath(dados_dir)}')

################## Resumo final ##################

print(f'\n{"="*60}')
print(f'[OK] Todas as {len(VARIACOES)} variacoes processadas com sucesso!')
print(f'Dados salvos em: {os.path.abspath(dados_dir_raiz)}')
print(f'{"="*60}')
