# -*- coding: utf-8 -*-
"""
Regressao com Random Forest
Dataset: ds_salaries.csv
Target: salary_in_usd

Testa cada configuracao com e sem log transform no target,
pois Random Forest e invariante a escala e pode performar
melhor com o target na escala original.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
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

dados_dir_raiz  = os.path.join(_dir, '..', 'dados_processados')
graficos_dir    = os.path.join(_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

caminho_resultados  = os.path.join(_dir, 'resultados_random_forest.csv')
caminho_consolidado = os.path.join(_dir, '..', 'resultados_comparativos', 'melhores_resultados.csv')

if not os.path.exists(dados_dir_raiz):
    raise FileNotFoundError('Pasta "dados_processados" nao encontrada!\n'
                            'Execute primeiro o script de pre-processamento unificado.')

################## Configuracao do grid ##################

N_ESTIMATORS_LIST  = [100, 200, 300]
MAX_DEPTHS         = [10, 20, None]
MAX_FEATURES_LIST  = ['sqrt', 'log2', None]
TARGET_TRANSFORMS  = ['nenhum', 'log']

################## Detectar bases ##################

bases = []
for item in sorted(os.listdir(dados_dir_raiz)):
    pasta = os.path.join(dados_dir_raiz, item)
    if os.path.isdir(pasta) and os.path.exists(os.path.join(pasta, 'config.json')):
        bases.append(pasta)

if not bases:
    raise FileNotFoundError('Nenhuma base encontrada em dados_processados/.')

total_configs = len(N_ESTIMATORS_LIST) * len(MAX_DEPTHS) * len(MAX_FEATURES_LIST) * len(TARGET_TRANSFORMS)
print(f'Bases encontradas: {len(bases)}')
print(f'Configuracoes por base: {total_configs}')

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
    y_treino = pd.read_csv(os.path.join(dados_dir, 'objetivo_treinamento.csv'))
    y_teste  = pd.read_csv(os.path.join(dados_dir, 'objetivo_teste.csv'))

    y_treino_orig = y_treino.values.ravel()
    y_teste_orig  = y_teste.values.ravel()

    n_features = X_treino.shape[1]

    for target_tf in TARGET_TRANSFORMS:
        if target_tf == 'log':
            y_treino_t = np.log1p(y_treino_orig)
        else:
            y_treino_t = y_treino_orig.copy()

        for n_est in N_ESTIMATORS_LIST:
            for depth in MAX_DEPTHS:
                for max_feat in MAX_FEATURES_LIST:
                    feat_label  = max_feat if max_feat is not None else 'all'
                    depth_label = depth if depth is not None else 'None'

                    regressor = RandomForestRegressor(
                        n_estimators=n_est,
                        max_features=max_feat,
                        max_depth=depth,
                        random_state=0
                    )
                    regressor.fit(X_treino, y_treino_t)
                    previsoes_raw = regressor.predict(X_teste)

                    if target_tf == 'log':
                        previsoes_orig = np.expm1(previsoes_raw)
                    else:
                        previsoes_orig = previsoes_raw

                    r2   = metrics.r2_score(y_teste_orig, previsoes_orig)
                    mae_ = metrics.mean_absolute_error(y_teste_orig, previsoes_orig)
                    mse_ = metrics.mean_squared_error(y_teste_orig, previsoes_orig)
                    rmse_= np.sqrt(mse_)
                    mape_= mape(y_teste_orig, previsoes_orig)

                    print(f'  [{target_tf}] n_est={n_est}, depth={depth_label}, feat={feat_label} | R2={r2:.4f} | MAPE={mape_:.1f}%')

                    todos_resultados.append({
                        'preprocessamento': NOME,
                        'encoding':         encoding,
                        'padronizacao':     padronizacao,
                        'target_transform': target_tf,
                        'n_estimators':     n_est,
                        'max_depth':        depth_label,
                        'max_features':     feat_label,
                        'r2_score':         round(r2,    6),
                        'mape':             round(mape_, 2),
                        'mae':              round(mae_,  2),
                        'mse':              round(mse_,  2),
                        'rmse':             round(rmse_, 2),
                        '_previsoes':      previsoes_orig,
                        '_y_teste_orig':   y_teste_orig,
                        '_X_treino':       X_treino,
                        '_regressor':      regressor,
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

nome_melhor  = melhor_full['preprocessamento']
n_est_melhor = melhor_full['n_estimators']
depth_melhor = melhor_full['max_depth']
feat_melhor  = melhor_full['max_features']
tf_melhor    = melhor_full['target_transform']
print(f'\n[MELHOR] {nome_melhor} n_est={n_est_melhor} depth={depth_melhor} feat={feat_melhor} target={tf_melhor}')
print(f'  R2={melhor_row["r2_score"]:.4f}')

previsoes_melhor = melhor_full['_previsoes']
y_teste_melhor   = melhor_full['_y_teste_orig']
X_treino_melhor  = melhor_full['_X_treino']
reg_melhor       = melhor_full['_regressor']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Random Forest - Melhor Resultado\n{nome_melhor} | n_est={n_est_melhor}, depth={depth_melhor}, feat={feat_melhor}, target={tf_melhor} | R2={melhor_row["r2_score"]:.4f}',
             fontsize=12, fontweight='bold')

axes[0].scatter(y_teste_melhor, previsoes_melhor, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.4)
lim = [y_teste_melhor.min(), y_teste_melhor.max()]
axes[0].plot(lim, lim, color='red', linestyle='--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel('Salario Real (USD)')
axes[0].set_ylabel('Salario Previsto (USD)')
axes[0].set_title('Real vs Previsto')
axes[0].legend()

importancias = reg_melhor.feature_importances_
n_top = min(15, len(importancias))
indices_top  = np.argsort(importancias)[-n_top:]
axes[1].barh(range(len(indices_top)), importancias[indices_top], align='center', color='seagreen', alpha=0.8)
axes[1].set_yticks(range(len(indices_top)))
axes[1].set_yticklabels([X_treino_melhor.columns[i] for i in indices_top], fontsize=8)
axes[1].set_xlabel('Importancia')
axes[1].set_title(f'Top {n_top} Features')

plt.tight_layout()
caminho_plot = os.path.join(graficos_dir, f'melhor_{nome_melhor}_est{n_est_melhor}_d{depth_melhor}_f{feat_melhor}_{tf_melhor}.png')
plt.savefig(caminho_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Plot salvo em: {caminho_plot}')

################## ETAPA 3: Append no arquivo consolidado ##################

os.makedirs(os.path.dirname(caminho_consolidado), exist_ok=True)

nova_linha = pd.DataFrame([{
    'metodo':           'Random Forest',
    'preprocessamento': melhor_row['preprocessamento'],
    'encoding':         melhor_row['encoding'],
    'padronizacao':     melhor_row['padronizacao'],
    'configuracao':     f'n_est={n_est_melhor}, depth={depth_melhor}, feat={feat_melhor}, target={tf_melhor}',
    'r2_score':         melhor_row['r2_score'],
    'mape':             melhor_row['mape'],
    'mae':              melhor_row['mae'],
    'mse':              melhor_row['mse'],
    'rmse':             melhor_row['rmse'],
}])

if os.path.exists(caminho_consolidado):
    df_cons = pd.read_csv(caminho_consolidado)
    df_cons = df_cons[df_cons['metodo'] != 'Random Forest']
    df_cons = pd.concat([df_cons, nova_linha], ignore_index=True)
else:
    df_cons = nova_linha

df_cons.to_csv(caminho_consolidado, index=False)
print(f'[OK] Melhor resultado adicionado em: {caminho_consolidado}')

################## Resumo final ##################

print(f'\n{"="*60}')
print(f'[OK] Random Forest concluido!')
print(f'  Total de testes: {len(todos_resultados)}')
print(f'  Melhor R2: {melhor_row["r2_score"]:.4f} ({nome_melhor}, n_est={n_est_melhor}, depth={depth_melhor}, feat={feat_melhor}, target={tf_melhor})')
print(f'{"="*60}')
