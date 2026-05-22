# -*- coding: utf-8 -*-
"""
Regressão Linear Múltipla
Dataset: ds_salaries.csv
Target: salary_in_usd
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo não-interativo (apenas salva gráficos)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import json

################## Configuração de diretórios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')

# Verificar se os dados processados existem
if not os.path.exists(dados_dir_raiz):
    raise FileNotFoundError(
        'Pasta "dados_processados" não encontrada!\n'
        'Execute primeiro o script de pré-processamento unificado.'
    )

# Pasta para salvar os gráficos
graficos_dir = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

# Caminho do arquivo de resultados
caminho_resultados = os.path.join(_dir, 'resultados_regressao_linear.csv')

################## Detectar todas as bases pré-processadas ##################

# Cada subpasta com config.json é uma base válida
bases = []
for item in sorted(os.listdir(dados_dir_raiz)):
    pasta = os.path.join(dados_dir_raiz, item)
    if os.path.isdir(pasta) and os.path.exists(os.path.join(pasta, 'config.json')):
        bases.append(pasta)

if not bases:
    raise FileNotFoundError(
        'Nenhuma base pré-processada encontrada em dados_processados/.\n'
        'Execute primeiro o script de pré-processamento unificado.'
    )

print(f'Bases encontradas: {len(bases)}')

################## Lista para acumular resultados ##################

todos_resultados = []

################## Loop por cada base pré-processada ##################

for dados_dir in bases:
    # Carregar configuração
    with open(os.path.join(dados_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    NOME_PREPROCESSAMENTO = config['NOME_PREPROCESSAMENTO']

    print(f'\n{"="*60}')
    print(f'  Regressao Linear - {NOME_PREPROCESSAMENTO}')
    print(f'{"="*60}')

    # Carregar datasets
    previsores_treinamento = pd.read_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'))
    previsores_teste = pd.read_csv(os.path.join(dados_dir, 'previsores_teste.csv'))
    objetivo_treinamento = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'))
    objetivo_teste = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv'))

    ################## Regressão Linear Múltipla ##################

    regressor = LinearRegression()

    # Treinamento
    regressor.fit(previsores_treinamento, objetivo_treinamento)

    # Teste
    previsoes = regressor.predict(previsores_teste)

    ################## Avaliação dos resultados ##################

    score = regressor.score(previsores_teste, objetivo_teste)
    mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
    mse = metrics.mean_squared_error(objetivo_teste, previsoes)
    rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

    print(f'Score (R2): {score}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    # Parâmetros estimados para o modelo
    coef_0 = regressor.intercept_
    coeficientes = regressor.coef_

    ################## Visualizações - Salvando como imagem ##################

    nome_grafico = NOME_PREPROCESSAMENTO

    # Valores reais vs previstos
    plt.figure(figsize=(10, 6))
    plt.scatter(objetivo_teste.values, previsoes, alpha=0.5)
    plt.plot([objetivo_teste.values.min(), objetivo_teste.values.max()],
             [objetivo_teste.values.min(), objetivo_teste.values.max()],
             color='red', linestyle='--')
    plt.title('Regressão Linear - Valores Reais vs Previstos')
    plt.xlabel('Salário Real (USD)')
    plt.ylabel('Salário Previsto (USD)')
    caminho_grafico_reais = os.path.join(graficos_dir, f'{nome_grafico}.png')
    plt.savefig(caminho_grafico_reais, dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfico de resíduos
    residuos = objetivo_teste.values.flatten() - previsoes.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(previsoes, residuos, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Regressão Linear - Gráfico de Resíduos')
    plt.xlabel('Valores Previstos (USD)')
    plt.ylabel('Resíduos')
    caminho_grafico_residuos = os.path.join(graficos_dir, f'{nome_grafico}_residuos.png')
    plt.savefig(caminho_grafico_residuos, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Graficos salvos em: {graficos_dir}')

    ################## Registro dos resultados ##################

    # Determinando tipo de encoding e padronização a partir do NOME_PREPROCESSAMENTO
    if 'com_dummy' in NOME_PREPROCESSAMENTO:
        encoding = 'one_hot'
    else:
        encoding = 'label_encoder'

    if 'com_std' in NOME_PREPROCESSAMENTO:
        padronizacao = 'sim'
    else:
        padronizacao = 'nao'

    resultado = {
        'preprocessamento': NOME_PREPROCESSAMENTO,
        'encoding': encoding,
        'score_r2': round(score, 6),
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'rmse': round(rmse, 2)
    }

    todos_resultados.append(resultado)

################## Salvando todos os resultados de uma vez ##################

df_resultados = pd.DataFrame(todos_resultados)
df_resultados.to_csv(caminho_resultados, index=False)

print(f'\n{"="*60}')
print(f'[OK] Regressao Linear concluida para {len(todos_resultados)} bases!')
print(f'Resultados salvos em: {caminho_resultados}')
print(f'Graficos salvos em: {graficos_dir}')
print(f'{"="*60}')
print(f'\n{df_resultados.to_string(index=False)}')
