# -*- coding: utf-8 -*-
"""
Regressão Polinomial
Dataset: ds_salaries.csv
Target: salary_in_usd

Carrega automaticamente os dados do último pré-processamento executado.
Basta rodar um dos scripts de pré-processamento e depois executar este arquivo.

Testa automaticamente graus de 2 a 7, registrando cada resultado no CSV
e salvando os gráficos como imagem.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import json

################## Carregando dados do pré-processamento ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

dados_dir = os.path.join(_dir, '..', 'dados_processados')

# Verificar se os dados processados existem
if not os.path.exists(dados_dir):
    raise FileNotFoundError(
        'Pasta "dados_processados" não encontrada!\n'
        'Execute primeiro um dos scripts de pré-processamento.'
    )

# Carregar configuração
with open(os.path.join(dados_dir, 'config.json'), 'r') as f:
    config = json.load(f)

NOME_PREPROCESSAMENTO = config['NOME_PREPROCESSAMENTO']
print(f'Pré-processamento carregado: {NOME_PREPROCESSAMENTO}')

# Carregar datasets
previsores_treinamento = pd.read_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'))
previsores_teste = pd.read_csv(os.path.join(dados_dir, 'previsores_teste.csv'))
objetivo_treinamento = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'))
objetivo_teste = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv'))

################## Configuração ##################

GRAUS = range(2, 8)  # Testa graus de 2 a 7

# Pasta para salvar os gráficos
graficos_dir = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

# Caminho do arquivo de resultados
caminho_resultados = os.path.join(_dir, 'resultados_regressao_polinomial.csv')

# Determinando tipo de encoding e padronização a partir do NOME_PREPROCESSAMENTO
if 'com_dummy' in NOME_PREPROCESSAMENTO:
    encoding = 'one_hot'
else:
    encoding = 'label_encoder'

if 'com_std' in NOME_PREPROCESSAMENTO:
    padronizacao = 'sim'
else:
    padronizacao = 'nao'

################## Loop por cada grau ##################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

for grau in GRAUS:
    print(f'\n{"="*60}')
    print(f'  Testando grau {grau}')
    print(f'{"="*60}')

    ################## Regressão Polinomial ##################

    poly = PolynomialFeatures(degree=grau)
    previsores_treinamento_poly = poly.fit_transform(previsores_treinamento)
    previsores_teste_poly = poly.transform(previsores_teste)

    regressor = LinearRegression()

    # Treinamento
    regressor.fit(previsores_treinamento_poly, objetivo_treinamento)

    score_treino = regressor.score(previsores_treinamento_poly, objetivo_treinamento)

    # Teste
    previsoes = regressor.predict(previsores_teste_poly)

    ################## Avaliação dos resultados ##################

    score = regressor.score(previsores_teste_poly, objetivo_teste)
    mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
    mse = metrics.mean_squared_error(objetivo_teste, previsoes)
    rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

    print(f'Grau do polinômio: {grau}')
    print(f'Score (R²) Treinamento: {score_treino}')
    print(f'Score (R²) Teste: {score}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    # Parâmetros estimados para o modelo
    coef_0 = regressor.intercept_
    coeficientes = regressor.coef_

    ################## Visualizações - Salvando como imagem ##################

    nome_grafico = f'{NOME_PREPROCESSAMENTO}_grau_{grau}'

    # Valores reais vs previstos
    plt.figure(figsize=(10, 6))
    plt.scatter(objetivo_teste.values, previsoes, alpha=0.5)
    plt.plot([objetivo_teste.values.min(), objetivo_teste.values.max()],
             [objetivo_teste.values.min(), objetivo_teste.values.max()],
             color='red', linestyle='--')
    plt.title(f'Regressão Polinomial (grau={grau}) - Valores Reais vs Previstos')
    plt.xlabel('Salário Real (USD)')
    plt.ylabel('Salário Previsto (USD)')
    caminho_grafico_reais = os.path.join(graficos_dir, f'{nome_grafico}.png')
    plt.savefig(caminho_grafico_reais, dpi=150, bbox_inches='tight')
    plt.show()

    # Gráfico de resíduos
    residuos = objetivo_teste.values.flatten() - previsoes.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(previsoes, residuos, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Regressão Polinomial (grau={grau}) - Gráfico de Resíduos')
    plt.xlabel('Valores Previstos (USD)')
    plt.ylabel('Resíduos')
    caminho_grafico_residuos = os.path.join(graficos_dir, f'{nome_grafico}_residuos.png')
    plt.savefig(caminho_grafico_residuos, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Gráficos salvos em: {graficos_dir}/{nome_grafico}_*.png')

    ################## Registro dos resultados em CSV ##################

    resultado = {
        'preprocessamento': NOME_PREPROCESSAMENTO,
        'encoding': encoding,
        'grau': grau,
        'score_r2': round(score, 6),
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'rmse': round(rmse, 2)
    }

    # Append no CSV (cria cabeçalho se não existir)
    df_resultado = pd.DataFrame([resultado])
    if os.path.exists(caminho_resultados):
        df_resultado.to_csv(caminho_resultados, mode='a', header=False, index=False)
    else:
        df_resultado.to_csv(caminho_resultados, mode='w', header=True, index=False)

    print(f'Resultado grau {grau} salvo em: {caminho_resultados}')
    print(df_resultado.to_string(index=False))

print(f'\n{"="*60}')
print(f'✅ Todos os graus testados ({list(GRAUS)})')
print(f'Resultados em: {caminho_resultados}')
print(f'Gráficos em: {graficos_dir}')
print(f'{"="*60}')
