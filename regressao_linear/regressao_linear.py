# -*- coding: utf-8 -*-
"""
Regressao Linear Multipla
Dataset: ds_salaries.csv
Target: salary_in_usd (log-transformado para treinamento)

Melhorias:
  - Log transform do target (np.log1p) para reduzir impacto de outliers
  - Metricas computadas na escala original (USD) e log
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import json

def mape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

################## Configuracao de diretorios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')
graficos_dir   = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

caminho_resultados   = os.path.join(_dir, 'resultados_regressao_linear.csv')
caminho_consolidado  = os.path.join(_dir, '..', 'resultados_comparativos', 'melhores_resultados.csv')

if not os.path.exists(dados_dir_raiz):
    raise FileNotFoundError('Pasta "dados_processados" nao encontrada!\n'
                            'Execute primeiro o script de pre-processamento unificado.')

################## Detectar bases pre-processadas ##################

bases = []
for item in sorted(os.listdir(dados_dir_raiz)):
    pasta = os.path.join(dados_dir_raiz, item)
    if os.path.isdir(pasta) and os.path.exists(os.path.join(pasta, 'config.json')):
        bases.append(pasta)

if not bases:
    raise FileNotFoundError('Nenhuma base encontrada em dados_processados/.')

print(f'Bases encontradas: {len(bases)}')

################## ETAPA 1: Rodar modelo e preencher CSV ##################

todos_resultados = []

for dados_dir in bases:
    with open(os.path.join(dados_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    NOME = config['NOME_PREPROCESSAMENTO']
    encoding     = config.get('encoding', 'N/A')
    padronizacao = config.get('padronizacao', 'N/A')

    print(f'\n{"="*60}')
    print(f'  Regressao Linear - {NOME}')
    print(f'{"="*60}')

    X_treino = pd.read_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'))
    X_teste  = pd.read_csv(os.path.join(dados_dir, 'previsores_teste.csv'))
    y_treino = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'))
    y_teste  = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv'))

    # Log transform do target
    y_treino_log = np.log1p(y_treino.values.ravel())
    y_teste_log  = np.log1p(y_teste.values.ravel())
    y_teste_orig = y_teste.values.ravel()

    regressor = LinearRegression()
    regressor.fit(X_treino, y_treino_log)
    previsoes_log = regressor.predict(X_teste)

    # Metricas no espaco log
    r2_log = metrics.r2_score(y_teste_log, previsoes_log)

    # Converter para escala original
    previsoes_orig = np.expm1(previsoes_log)

    # Metricas na escala original (USD)
    r2_orig = metrics.r2_score(y_teste_orig, previsoes_orig)
    mae_ = metrics.mean_absolute_error(y_teste_orig, previsoes_orig)
    mse_ = metrics.mean_squared_error(y_teste_orig, previsoes_orig)
    rmse_= np.sqrt(mse_)
    mape_= mape(y_teste_orig, previsoes_orig)

    print(f'  R2:     {r2_orig:.6f}')
    print(f'  R2_log: {r2_log:.6f}')
    print(f'  MAPE:   {mape_:.2f}%')
    print(f'  MAE:    {mae_:.2f}')
    print(f'  RMSE:   {rmse_:.2f}')

    todos_resultados.append({
        'preprocessamento': NOME,
        'encoding':         encoding,
        'padronizacao':     padronizacao,
        'r2_score':         round(r2_orig, 6),
        'r2_log':           round(r2_log,  6),
        'mape':             round(mape_, 2),
        'mae':              round(mae_,  2),
        'mse':              round(mse_,  2),
        'rmse':             round(rmse_, 2),
        # armazenar para o plot (nao vai para o CSV)
        '_dados_dir': dados_dir,
        '_previsoes': previsoes_orig,
        '_y_teste':   y_teste_orig,
    })

# Salva CSV sem as colunas internas (_)
df_resultados = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                               for r in todos_resultados])
df_resultados.to_csv(caminho_resultados, index=False)
print(f'\n[OK] Resultados salvos em: {caminho_resultados}')

################## ETAPA 2: Analisar CSV e gerar plot do MELHOR ##################

melhor_idx  = df_resultados['r2_score'].idxmax()
melhor_row  = df_resultados.loc[melhor_idx]
melhor_full = todos_resultados[melhor_idx]

print(f'\n[MELHOR] {melhor_full["preprocessamento"]} | R2={melhor_row["r2_score"]:.4f} | R2_log={melhor_row["r2_log"]:.4f}')

previsoes_melhor = melhor_full['_previsoes']
y_teste_melhor   = melhor_full['_y_teste']
nome_melhor      = melhor_full['preprocessamento']

# Plot: Valores Reais vs Previstos
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Regressao Linear - Melhor Resultado\n{nome_melhor} | R2={melhor_row["r2_score"]:.4f}',
             fontsize=13, fontweight='bold')

ax.scatter(y_teste_melhor, previsoes_melhor, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.4)
lim = [y_teste_melhor.min(), y_teste_melhor.max()]
ax.plot(lim, lim, color='red', linestyle='--', linewidth=1.5, label='Ideal')
ax.set_xlabel('Salario Real (USD)')
ax.set_ylabel('Salario Previsto (USD)')
ax.set_title('Real vs Previsto')
ax.legend()

plt.tight_layout()
caminho_plot = os.path.join(graficos_dir, f'melhor_{nome_melhor}.png')
plt.savefig(caminho_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Plot salvo em: {caminho_plot}')

################## ETAPA 3: Append no arquivo consolidado ##################

os.makedirs(os.path.dirname(caminho_consolidado), exist_ok=True)

nova_linha = pd.DataFrame([{
    'metodo':           'Regressao Linear',
    'preprocessamento': melhor_row['preprocessamento'],
    'encoding':         melhor_row['encoding'],
    'padronizacao':     melhor_row['padronizacao'],
    'configuracao':     'N/A',
    'r2_score':         melhor_row['r2_score'],
    'r2_log':           melhor_row['r2_log'],
    'mape':             melhor_row['mape'],
    'mae':              melhor_row['mae'],
    'mse':              melhor_row['mse'],
    'rmse':             melhor_row['rmse'],
}])

if os.path.exists(caminho_consolidado):
    df_cons = pd.read_csv(caminho_consolidado)
    # Remove linha anterior do mesmo modelo se existir
    df_cons = df_cons[df_cons['metodo'] != 'Regressao Linear']
    df_cons = pd.concat([df_cons, nova_linha], ignore_index=True)
else:
    df_cons = nova_linha

df_cons.to_csv(caminho_consolidado, index=False)
print(f'[OK] Melhor resultado adicionado em: {caminho_consolidado}')

################## Resumo final ##################

print(f'\n{"="*60}')
print(f'[OK] Regressao Linear concluida!')
print(f'  Total de bases testadas: {len(bases)}')
print(f'  Melhor R2: {melhor_row["r2_score"]:.4f} ({nome_melhor})')
print(f'  Melhor R2_log: {melhor_row["r2_log"]:.4f}')
print(f'{"="*60}')
print(f'\n{df_resultados.to_string(index=False)}')
