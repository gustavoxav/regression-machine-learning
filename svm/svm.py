# -*- coding: utf-8 -*-
"""
Regressao com SVM (Support Vector Machine)
Dataset: ds_salaries.csv
Target: salary_in_usd
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
import os
import json


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


################## Configuracao de diretorios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')

if not os.path.exists(dados_dir_raiz):
    raise FileNotFoundError(
        'Pasta "dados_processados" nao encontrada!\n'
        'Execute primeiro o script de pre-processamento unificado.'
    )

graficos_dir = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

caminho_resultados = os.path.join(_dir, 'resultados_svm.csv')

################## Configuracao ##################

CONFIGURACOES_SVM = [
    {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1},
    {'kernel': 'linear', 'C': 10.0, 'epsilon': 0.1},
    {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 100.0, 'epsilon': 0.1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 100.0, 'epsilon': 0.2, 'gamma': 'auto'},
    {'kernel': 'poly', 'C': 10.0, 'epsilon': 0.1, 'gamma': 'scale', 'degree': 2}
]

################## Detectar todas as bases pre-processadas ##################

bases = []
for item in sorted(os.listdir(dados_dir_raiz)):
    pasta = os.path.join(dados_dir_raiz, item)
    if os.path.isdir(pasta) and os.path.exists(os.path.join(pasta, 'config.json')):
        bases.append(pasta)

if not bases:
    raise FileNotFoundError(
        'Nenhuma base pre-processada encontrada em dados_processados/.\n'
        'Execute primeiro o script de pre-processamento unificado.'
    )

print(f'Bases encontradas: {len(bases)}')
print(f'Configuracoes por base: {len(CONFIGURACOES_SVM)}')

################## Lista para acumular resultados ##################

todos_resultados = []

################## Loop por cada base pre-processada ##################

for dados_dir in bases:
    with open(os.path.join(dados_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    NOME_PREPROCESSAMENTO = config['NOME_PREPROCESSAMENTO']

    print(f'\n{"#"*60}')
    print(f'  Base: {NOME_PREPROCESSAMENTO}')
    print(f'{"#"*60}')

    previsores_treinamento = pd.read_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'))
    previsores_teste = pd.read_csv(os.path.join(dados_dir, 'previsores_teste.csv'))
    objetivo_treinamento = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'))
    objetivo_teste = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv'))

    y_treino = objetivo_treinamento.values.ravel()
    y_teste = objetivo_teste.values.ravel()

    if 'com_dummy' in NOME_PREPROCESSAMENTO:
        encoding = 'one_hot'
    else:
        encoding = 'label_encoder'

    for idx, cfg in enumerate(CONFIGURACOES_SVM, start=1):
        print(f'\n{"="*60}')
        print(f'  Testando configuracao {idx}: {cfg} ({NOME_PREPROCESSAMENTO})')
        print(f'{"="*60}')

        regressor = SVR(**cfg)
        regressor.fit(previsores_treinamento, y_treino)

        score_treino = regressor.score(previsores_treinamento, y_treino)
        previsoes = regressor.predict(previsores_teste)

        score = regressor.score(previsores_teste, y_teste)
        mae = metrics.mean_absolute_error(y_teste, previsoes)
        mse = metrics.mean_squared_error(y_teste, previsoes)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_teste, previsoes)

        print(f'Score (R2) Treinamento: {score_treino}')
        print(f'Score (R2) Teste: {score}')
        print(f'Mean Absolute Percentage Error: {mape}')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')

        nome_cfg = (
            f"{NOME_PREPROCESSAMENTO}_"
            f"{cfg.get('kernel', 'na')}_"
            f"C_{cfg.get('C', 'na')}_"
            f"eps_{cfg.get('epsilon', 'na')}_"
            f"g_{cfg.get('gamma', 'na')}_"
            f"d_{cfg.get('degree', 'na')}"
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(y_teste, previsoes, alpha=0.5)
        plt.plot([y_teste.min(), y_teste.max()],
                 [y_teste.min(), y_teste.max()],
                 color='red', linestyle='--')
        plt.title('SVM - Valores Reais vs Previstos')
        plt.xlabel('Salario Real (USD)')
        plt.ylabel('Salario Previsto (USD)')
        caminho_grafico_reais = os.path.join(graficos_dir, f'{nome_cfg}.png')
        plt.savefig(caminho_grafico_reais, dpi=150, bbox_inches='tight')
        plt.close()

        residuos = y_teste - previsoes
        plt.figure(figsize=(10, 6))
        plt.scatter(previsoes, residuos, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('SVM - Grafico de Residuos')
        plt.xlabel('Valores Previstos (USD)')
        plt.ylabel('Residuos')
        caminho_grafico_residuos = os.path.join(graficos_dir, f'{nome_cfg}_residuos.png')
        plt.savefig(caminho_grafico_residuos, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'Graficos salvos em: {graficos_dir}/{nome_cfg}_*.png')

        resultado = {
            'preprocessamento': NOME_PREPROCESSAMENTO,
            'encoding': encoding,
            'kernel': cfg.get('kernel', None),
            'C': cfg.get('C', None),
            'epsilon': cfg.get('epsilon', None),
            'gamma': cfg.get('gamma', None),
            'degree': cfg.get('degree', None),
            'score_r2_treino': round(score_treino, 6),
            'score_r2': round(score, 6),
            'mape': round(mape, 2) if not np.isnan(mape) else None,
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2)
        }

        todos_resultados.append(resultado)

################## Salvando todos os resultados de uma vez ##################

df_resultados = pd.DataFrame(todos_resultados)
df_resultados.to_csv(caminho_resultados, index=False)

print(f'\n{"="*60}')
print(f'[OK] SVM concluida para {len(bases)} bases!')
print(f'Total de testes realizados: {len(todos_resultados)}')
print(f'Resultados salvos em: {caminho_resultados}')
print(f'Graficos salvos em: {graficos_dir}')
print(f'{"="*60}')
print(f'\n{df_resultados.to_string(index=False)}')
