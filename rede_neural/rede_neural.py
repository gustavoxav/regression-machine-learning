# -*- coding: utf-8 -*-
"""
Regressao com Rede Neural (MLPRegressor)
Dataset: ds_salaries.csv
Target: salary_in_usd

Fluxo:
  1. Roda todas as configuracoes em todas as bases -> preenche CSV
  2. Analisa o CSV e gera plot SOMENTE do melhor resultado (maior R2)
  3. Appenda o melhor resultado ao arquivo consolidado de melhores resultados
"""

import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import os
import json

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def mape_calc(y_true, y_pred):
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

dados_dir_raiz  = os.path.join(_dir, '..', 'dados_processados')
graficos_dir    = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

caminho_resultados  = os.path.join(_dir, 'resultados_rede_neural.csv')
caminho_consolidado = os.path.join(_dir, '..', 'resultados_comparativos', 'melhores_resultados.csv')

if not os.path.exists(dados_dir_raiz):
    raise FileNotFoundError('Pasta "dados_processados" nao encontrada!\n'
                            'Execute primeiro o script de pre-processamento unificado.')

################## Configuracao ##################

CONFIGURACOES_MLP = [
    {'hidden_layer_sizes': (32,),        'activation': 'relu', 'alpha': 0.0001},
    {'hidden_layer_sizes': (64,),        'activation': 'relu', 'alpha': 0.0001},
    {'hidden_layer_sizes': (128, 64),    'activation': 'relu', 'alpha': 0.0001},
    {'hidden_layer_sizes': (64,),        'activation': 'tanh', 'alpha': 0.0001},
    {'hidden_layer_sizes': (128, 64),    'activation': 'tanh', 'alpha': 0.001},
    {'hidden_layer_sizes': (128, 64, 32),'activation': 'relu', 'alpha': 0.001},
]

################## Detectar bases ##################

bases = []
for item in sorted(os.listdir(dados_dir_raiz)):
    pasta = os.path.join(dados_dir_raiz, item)
    if os.path.isdir(pasta) and os.path.exists(os.path.join(pasta, 'config.json')):
        bases.append(pasta)

if not bases:
    raise FileNotFoundError('Nenhuma base encontrada em dados_processados/.')

print(f'Bases encontradas: {len(bases)}')
print(f'Configuracoes por base: {len(CONFIGURACOES_MLP)}')

################## ETAPA 1: Rodar modelo e preencher CSV ##################

todos_resultados = []

for dados_dir in bases:
    with open(os.path.join(dados_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    NOME = config['NOME_PREPROCESSAMENTO']
    encoding     = config.get('encoding', 'N/A')
    padronizacao = config.get('padronizacao', 'N/A')

    print(f'\n{"#"*60}')
    print(f'  Base: {NOME}')
    print(f'{"#"*60}')

    X_treino = pd.read_csv(os.path.join(dados_dir, 'previsores_treinamento.csv'))
    X_teste  = pd.read_csv(os.path.join(dados_dir, 'previsores_teste.csv'))
    y_treino = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv')).values.ravel()
    y_teste  = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv')).values.ravel()

    for idx, cfg in enumerate(CONFIGURACOES_MLP, start=1):
        hl_str = '-'.join(str(x) for x in cfg['hidden_layer_sizes'])
        print(f'  [{idx}] hidden={hl_str}, act={cfg["activation"]}, alpha={cfg["alpha"]}...')

        regressor = MLPRegressor(
            hidden_layer_sizes=cfg['hidden_layer_sizes'],
            activation=cfg['activation'],
            alpha=cfg['alpha'],
            max_iter=1000,
            early_stopping=True,
            random_state=0
        )
        regressor.fit(X_treino, y_treino)
        previsoes = regressor.predict(X_teste)

        r2   = regressor.score(X_teste, y_teste)
        mae_ = metrics.mean_absolute_error(y_teste, previsoes)
        mse_ = metrics.mean_squared_error(y_teste, previsoes)
        rmse_= np.sqrt(mse_)
        mape_= mape_calc(y_teste, previsoes)

        print(f'    R2={r2:.4f} | MAPE={mape_:.2f}% | MAE={mae_:.0f} | RMSE={rmse_:.0f}')

        todos_resultados.append({
            'preprocessamento':   NOME,
            'encoding':           encoding,
            'padronizacao':       padronizacao,
            'hidden_layer_sizes': str(cfg['hidden_layer_sizes']),
            'activation':         cfg['activation'],
            'alpha':              cfg['alpha'],
            'r2_score':           round(r2,    6),
            'mape':               round(mape_, 2) if not np.isnan(mape_) else None,
            'mae':                round(mae_,  2),
            'mse':                round(mse_,  2),
            'rmse':               round(rmse_, 2),
            '_previsoes': previsoes,
            '_y_teste':   y_teste,
        })

# Salva CSV sem colunas internas
df_resultados = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                               for r in todos_resultados])
df_resultados.to_csv(caminho_resultados, index=False)
print(f'\n[OK] Resultados salvos em: {caminho_resultados}')

################## ETAPA 2: Analisar CSV e gerar plot do MELHOR ##################

melhor_idx  = df_resultados['r2_score'].idxmax()
melhor_row  = df_resultados.loc[melhor_idx]
melhor_full = todos_resultados[melhor_idx]

nome_melhor = melhor_full['preprocessamento']
cfg_melhor  = (f"hidden={melhor_full['hidden_layer_sizes']}, "
               f"act={melhor_full['activation']}, alpha={melhor_full['alpha']}")
print(f'\n[MELHOR] {nome_melhor} | {cfg_melhor} | R2={melhor_row["r2_score"]:.4f}')

previsoes_melhor = melhor_full['_previsoes']
y_teste_melhor   = melhor_full['_y_teste']

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Rede Neural - Melhor Resultado\n{nome_melhor} | R2={melhor_row["r2_score"]:.4f}',
             fontsize=13, fontweight='bold')

ax.scatter(y_teste_melhor, previsoes_melhor, alpha=0.6, color='steelblue',
           edgecolors='white', linewidth=0.4)
lim = [y_teste_melhor.min(), y_teste_melhor.max()]
ax.plot(lim, lim, color='red', linestyle='--', linewidth=1.5, label='Ideal')
ax.set_xlabel('Salario Real (USD)')
ax.set_ylabel('Salario Previsto (USD)')
ax.set_title('Real vs Previsto')
ax.legend()

plt.tight_layout()
caminho_plot = os.path.join(graficos_dir, f'melhor_{nome_melhor}_nn.png')
plt.savefig(caminho_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Plot salvo em: {caminho_plot}')

################## ETAPA 3: Append no arquivo consolidado ##################

os.makedirs(os.path.dirname(caminho_consolidado), exist_ok=True)

nova_linha = pd.DataFrame([{
    'modelo':           'Rede Neural',
    'preprocessamento': melhor_row['preprocessamento'],
    'encoding':         melhor_row['encoding'],
    'padronizacao':     melhor_row['padronizacao'],
    'configuracao':     cfg_melhor,
    'r2_score':         melhor_row['r2_score'],
    'mape':             melhor_row['mape'],
    'mae':              melhor_row['mae'],
    'mse':              melhor_row['mse'],
    'rmse':             melhor_row['rmse'],
}])

if os.path.exists(caminho_consolidado):
    df_cons = pd.read_csv(caminho_consolidado)
    df_cons = df_cons[df_cons['modelo'] != 'Rede Neural']
    df_cons = pd.concat([df_cons, nova_linha], ignore_index=True)
else:
    df_cons = nova_linha

df_cons.to_csv(caminho_consolidado, index=False)
print(f'[OK] Melhor resultado adicionado em: {caminho_consolidado}')

################## Resumo final ##################

print(f'\n{"="*60}')
print(f'[OK] Rede Neural concluida!')
print(f'  Total de testes: {len(todos_resultados)}')
print(f'  Melhor R2: {melhor_row["r2_score"]:.4f} ({nome_melhor})')
print(f'{"="*60}')
print(f'\n{df_resultados.to_string(index=False)}')
