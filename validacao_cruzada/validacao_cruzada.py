"""
Validacao Cruzada - Avaliacao consolidada de todos os regressores

Regra principal (correcao do erro da P1):
  A padronizacao e realizada DENTRO do loop de cada fold, NAO antes.
  Ou seja, o StandardScaler recebe .fit() apenas nos dados de TREINO
  do fold atual, e .transform() e aplicado tanto no treino quanto no
  teste daquele fold. Isso evita data leakage.

Estrategia por modelo:
  - Modelos sensiveis a escala (SVM, Rede Neural, Reg. Linear, Reg. Polinomial):
      usar Pipeline(StandardScaler -> modelo), que o sklearn garante
      que o scaler so e fitado no fold de treino.
  - Modelos invariantes a escala (Arvore de Decisao, Random Forest):
      sem padronizacao necessaria, mas o Pipeline pode ser usado sem scaler.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

################## Configuracao de diretorios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

dados_dir_raiz = os.path.join(_dir, '..', 'dados_processados')
output_dir     = _dir
os.makedirs(output_dir, exist_ok=True)

N_FOLDS    = 5
RANDOM_STATE = 0

################## Funcoes auxiliares ##################

def mape_calc(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def carregar_base_sem_std(nome_base):
    """
    Carrega treino+teste da variacao SEM padronizacao e concatena para CV.
    O nome_base pode ser 'com_dummy_com_std' ou 'sem_dummy_com_std'; a funcao
    troca automaticamente para a variante sem_std correspondente.
    """
    # Garante que usamos a base sem padronizacao pre-aplicada
    base_sem_std = nome_base.replace('_com_std', '_sem_std')
    pasta = os.path.join(dados_dir_raiz, base_sem_std)

    if not os.path.exists(pasta):
        raise FileNotFoundError(
            f'Base sem padronizacao nao encontrada: {pasta}\n'
            f'Execute primeiro o pre-processamento unificado.'
        )

    X_tr = pd.read_csv(os.path.join(pasta, 'previsores_treinamento.csv'))
    X_te = pd.read_csv(os.path.join(pasta, 'previsores_teste.csv'))
    y_tr = pd.read_csv(os.path.join(pasta, 'objetivo_treinamento.csv')).iloc[:, 0]
    y_te = pd.read_csv(os.path.join(pasta, 'objetivo_teste.csv')).iloc[:, 0]

    X = pd.concat([X_tr, X_te], ignore_index=True).values.astype(float)
    y = pd.concat([y_tr, y_te], ignore_index=True).values.astype(float)

    return X, y, base_sem_std


def avaliar_fold(modelo_pipeline, X_tr, y_tr, X_te, y_te, target_tf):
    if target_tf == 'log':
        y_tr_t = np.log1p(y_tr)
        modelo_pipeline.fit(X_tr, y_tr_t)
        pred_raw = modelo_pipeline.predict(X_te)
        pred_raw = np.clip(pred_raw, -50, 50)
        pred = np.expm1(pred_raw)

    elif target_tf == 'std_target':
        # TransformedTargetRegressor envolve o pipeline e padroniza o target
        # TAMBEM apenas no fold de treino (fit interno ao .fit() do TTR)
        ttr = TransformedTargetRegressor(
            regressor=modelo_pipeline,
            transformer=StandardScaler()
        )
        ttr.fit(X_tr, y_tr)
        pred = ttr.predict(X_te)

    else:  # 'nenhum'
        modelo_pipeline.fit(X_tr, y_tr)
        pred = modelo_pipeline.predict(X_te)

    r2   = float(r2_score(y_te, pred))
    mae  = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mape = mape_calc(y_te, pred)

    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}


def rodar_cv(nome_modelo, base_nome, criar_pipeline_fn, target_tf):
    """
    Executa KFold CV para um modelo.
    A padronizacao e feita DENTRO do pipeline, DENTRO de cada fold.
    """
    X, y, base_real = carregar_base_sem_std(base_nome)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    metricas_fold = []
    print(f'\n{"="*70}')
    print(f'  {nome_modelo}')
    print(f'  Base: {base_real} | Target transform: {target_tf} | {N_FOLDS} folds')
    print(f'{"="*70}')

    for fold_idx, (idx_tr, idx_te) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]

        # *** PADRONIZACAO DENTRO DO FOLD ***
        # O pipeline chama scaler.fit(X_tr) e scaler.transform(X_tr, X_te)
        # internamente, sem nunca ver os dados de X_te no fit.
        pipeline = criar_pipeline_fn()
        fold_result = avaliar_fold(pipeline, X_tr, y_tr, X_te, y_te, target_tf)
        metricas_fold.append(fold_result)

        print(f'  Fold {fold_idx}: R2={fold_result["r2"]:+.4f} | '
              f'RMSE={fold_result["rmse"]:.0f} | '
              f'MAE={fold_result["mae"]:.0f} | '
              f'MAPE={fold_result["mape"]:.1f}%')

    r2_vals   = [m['r2']   for m in metricas_fold]
    mae_vals  = [m['mae']  for m in metricas_fold]
    rmse_vals = [m['rmse'] for m in metricas_fold]
    mape_vals = [m['mape'] for m in metricas_fold if not np.isnan(m['mape'])]

    resultado = {
        'modelo':          nome_modelo,
        'base':            base_real,
        'target_transform': target_tf,
        'r2_media':        round(float(np.mean(r2_vals)),   4),
        'r2_std':          round(float(np.std(r2_vals)),    4),
        'mae_media':       round(float(np.mean(mae_vals)),  2),
        'mae_std':         round(float(np.std(mae_vals)),   2),
        'rmse_media':      round(float(np.mean(rmse_vals)), 2),
        'rmse_std':        round(float(np.std(rmse_vals)),  2),
        'mape_media':      round(float(np.mean(mape_vals)), 2) if mape_vals else None,
        'mape_std':        round(float(np.std(mape_vals)),  2) if mape_vals else None,
        'r2_por_fold':     [round(v, 4) for v in r2_vals],
    }

    print(f'\n  >> R2: {resultado["r2_media"]:+.4f} +/- {resultado["r2_std"]:.4f}')
    print(f'  >> RMSE: {resultado["rmse_media"]:.0f} +/- {resultado["rmse_std"]:.0f}')
    print(f'  >> MAE:  {resultado["mae_media"]:.0f}  +/- {resultado["mae_std"]:.0f}')
    if resultado['mape_media'] is not None:
        print(f'  >> MAPE: {resultado["mape_media"]:.1f}% +/- {resultado["mape_std"]:.1f}%')

    return resultado


################## Definicao dos modelos ##################

def especificacoes_modelos():
    """
    Retorna lista de especificacoes dos modelos para CV.

    Pipeline com StandardScaler garante que o scaler so e fitado
    nos dados de treino de cada fold (correcao do professor).
    """
    return [
        # ------------------------------------------------------------------
        # 1. Regressao Linear
        #    Melhor base: com_dummy_sem_std (ja sem std) — sem scaler necessario
        #    mas incluimos para padronizacao consistente da metodologia.
        #    Target: log (treino com log1p, previsao com expm1)
        # ------------------------------------------------------------------
        {
            'nome':       'Regressao Linear',
            'base':       'com_dummy_sem_std',
            'target_tf':  'log',
            'criar_pipeline': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('reg',    LinearRegression()),
            ]),
        },

        # ------------------------------------------------------------------
        # 2. Regressao Polinomial
        #    Melhor base: com_dummy_com_std -> convertida para com_dummy_sem_std
        #    Grau 2, target: nenhum (melhor resultado da fase de selecao)
        # ------------------------------------------------------------------
        {
            'nome':       'Regressao Polinomial',
            'base':       'com_dummy_com_std',
            'target_tf':  'nenhum',
            'criar_pipeline': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
                ('reg',    LinearRegression()),
            ]),
        },

        # ------------------------------------------------------------------
        # 3. Arvore de Decisao
        #    Arvores sao invariantes a escala, mas incluimos scaler para
        #    uniformidade; nao causa nenhum dano.
        #    Melhor base: com_dummy_com_std -> com_dummy_sem_std
        #    max_depth=3, target: nenhum
        # ------------------------------------------------------------------
        {
            'nome':       'Arvore de Decisao',
            'base':       'com_dummy_com_std',
            'target_tf':  'nenhum',
            'criar_pipeline': lambda: Pipeline([
                ('reg', DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)),
            ]),
        },

        # ------------------------------------------------------------------
        # 4. Random Forest
        #    RF invariante a escala.
        #    Melhor base: sem_dummy_com_std -> sem_dummy_sem_std
        #    n_estimators=100, max_depth=10, max_features=sqrt, target: log
        # ------------------------------------------------------------------
        {
            'nome':       'Random Forest',
            'base':       'sem_dummy_com_std',
            'target_tf':  'log',
            'criar_pipeline': lambda: Pipeline([
                ('reg', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    max_features='sqrt',
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
        },

        # ------------------------------------------------------------------
        # 5. SVM
        #    MUITO sensivel a escala — scaler dentro do pipeline e essencial.
        #    Melhor base: com_dummy_com_std -> com_dummy_sem_std
        #    kernel=linear, C=10.0, epsilon=0.1, target: log
        # ------------------------------------------------------------------
        {
            'nome':       'SVM',
            'base':       'com_dummy_com_std',
            'target_tf':  'log',
            'criar_pipeline': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('svr',    SVR(kernel='linear', C=10.0, epsilon=0.1)),
            ]),
        },

        # ------------------------------------------------------------------
        # 6. Rede Neural (MLP)
        #    MUITO sensivel a escala — scaler dentro do pipeline e essencial.
        #    Melhor base: com_dummy_com_std -> com_dummy_sem_std
        #    hidden=(32,), activation=relu, alpha=0.0001, target: std_target
        #
        #    Para std_target: o TransformedTargetRegressor em avaliar_fold()
        #    envolve o pipeline e escala tambem o target apenas no fold de
        #    treino, sem vazamento para o fold de teste.
        # ------------------------------------------------------------------
        {
            'nome':       'Rede Neural',
            'base':       'com_dummy_com_std',
            'target_tf':  'std_target',
            'criar_pipeline': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('mlp',    MLPRegressor(
                    hidden_layer_sizes=(32,),
                    activation='relu',
                    alpha=0.0001,
                    max_iter=2000,
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                )),
            ]),
        },
    ]


################## Graficos ##################

def plotar_r2_comparacao(resultados):
    nomes  = [r['modelo'] for r in resultados]
    medias = [r['r2_media'] for r in resultados]
    desvios = [r['r2_std'] for r in resultados]

    cores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

    fig, ax = plt.subplots(figsize=(12, 6))
    barras = ax.bar(nomes, medias, yerr=desvios, capsize=6,
                    color=cores[:len(nomes)], alpha=0.85)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_ylabel('R² Médio (KFold CV)', fontsize=12)
    ax.set_title(
        f'Validação Cruzada — Comparação entre Regressores\n'
        f'(KFold, k={N_FOLDS}, padronização dentro de cada fold)',
        fontsize=13, fontweight='bold'
    )
    ax.set_ylim(min(min(medias) - 0.15, -0.1), max(max(medias) + 0.15, 0.7))

    for barra, media, desvio in zip(barras, medias, desvios):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + desvio + 0.01,
            f'{media:+.4f}\n±{desvio:.4f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold'
        )

    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=15, ha='right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    caminho = os.path.join(output_dir, 'comparacao_regressores_cv.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] Grafico salvo em: {caminho}')


def plotar_rmse_comparacao(resultados):
    nomes  = [r['modelo'] for r in resultados]
    medias = [r['rmse_media'] for r in resultados]
    desvios = [r['rmse_std'] for r in resultados]

    cores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(nomes, medias, yerr=desvios, capsize=6,
           color=cores[:len(nomes)], alpha=0.85)
    ax.set_ylabel('RMSE Médio (USD)', fontsize=12)
    ax.set_title(
        f'Validação Cruzada — RMSE por Regressor\n'
        f'(KFold, k={N_FOLDS}, padronização dentro de cada fold)',
        fontsize=13, fontweight='bold'
    )

    for i, (media, desvio) in enumerate(zip(medias, desvios)):
        ax.text(i, media + desvio + 500, f'{media:.0f}\n±{desvio:.0f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=15, ha='right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    caminho = os.path.join(output_dir, 'comparacao_rmse_cv.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] Grafico RMSE salvo em: {caminho}')


################## MAIN ##################

def main():
    print('\n' + '=' * 70)
    print('  Validacao Cruzada — Regressao (KFold, k=5)')
    print('  CORRECAO P1: padronizacao realizada DENTRO de cada fold')
    print('  (scaler.fit() apenas no X_treino do fold, sem data leakage)')
    print('=' * 70)

    especificacoes = especificacoes_modelos()
    resultados = []

    for esp in especificacoes:
        resultado = rodar_cv(
            nome_modelo      = esp['nome'],
            base_nome        = esp['base'],
            criar_pipeline_fn= esp['criar_pipeline'],
            target_tf        = esp['target_tf'],
        )
        resultados.append(resultado)

    # --- Salvar CSV ---
    linhas_csv = []
    for r in resultados:
        linhas_csv.append({
            'modelo':          r['modelo'],
            'base':            r['base'],
            'target_transform': r['target_transform'],
            'r2_media':        r['r2_media'],
            'r2_std':          r['r2_std'],
            'mae_media':       r['mae_media'],
            'mae_std':         r['mae_std'],
            'rmse_media':      r['rmse_media'],
            'rmse_std':        r['rmse_std'],
            'mape_media':      r['mape_media'],
            'mape_std':        r['mape_std'],
            'r2_por_fold':     str(r['r2_por_fold']),
        })

    df = pd.DataFrame(linhas_csv).sort_values('r2_media', ascending=False)
    caminho_csv = os.path.join(output_dir, 'resultados_cv.csv')
    df.to_csv(caminho_csv, index=False)
    print(f'\n[OK] CSV salvo em: {caminho_csv}')

    # --- Salvar JSON ---
    caminho_json = os.path.join(output_dir, 'resumo_cv.json')
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f'[OK] JSON salvo em: {caminho_json}')

    # --- Graficos ---
    plotar_r2_comparacao(resultados)
    plotar_rmse_comparacao(resultados)

    # --- Ranking final ---
    print(f'\n{"=" * 70}')
    print(f'  RANKING FINAL — Validacao Cruzada (por R² médio)')
    print(f'{"=" * 70}')
    for _, row in df.iterrows():
        print(
            f'  {row["modelo"]:22s} | Base: {row["base"]:22s} '
            f'| R2: {row["r2_media"]:+.4f} +/- {row["r2_std"]:.4f} '
            f'| RMSE: {row["rmse_media"]:.0f} +/- {row["rmse_std"]:.0f}'
        )
    print(f'{"=" * 70}\n')


if __name__ == '__main__':
    main()
