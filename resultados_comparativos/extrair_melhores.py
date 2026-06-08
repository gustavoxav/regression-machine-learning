# -*- coding: utf-8 -*-
"""
Comparador e Extrator dos Melhores Resultados
Cria um resumo consolidado em CSV e JSON com o melhor resultado de cada modelo.
"""

import pandas as pd
import json
import os

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

# Mapeamento dos caminhos dos resultados
CAMINHOS_RESULTADOS = {
    'Regressao Linear': os.path.join(_dir, '..', 'regressao_linear', 'resultados_regressao_linear.csv'),
    'Regressao Polinomial': os.path.join(_dir, '..', 'regressao_polinomial', 'resultados_regressao_polinomial.csv'),
    'Arvore de Decisao': os.path.join(_dir, '..', 'arvore_decisao', 'resultados_arvore_decisao.csv'),
    'Random Forest': os.path.join(_dir, '..', 'random_forest', 'resultados_random_forest.csv'),
    'SVM': os.path.join(_dir, '..', 'svm', 'resultados_svm.csv'),
    'Rede Neural': os.path.join(_dir, '..', 'rede_neural', 'resultados_rede_neural.csv')
}

caminho_csv_saida = os.path.join(_dir, 'melhores_resultados.csv')
caminho_json_saida = os.path.join(_dir, 'melhores_resultados.json')

# Colunas de metricas (nao sao hiperparametros)
COLUNAS_METRICAS = ['r2_score', 'r2_log', 'score_r2', 'score_r2_treino', 'mape', 'mae', 'mse', 'rmse']
COLUNAS_BASE = ['preprocessamento', 'encoding', 'padronizacao']

def extrair_melhores():
    melhores = []
    
    for metodo, caminho in CAMINHOS_RESULTADOS.items():
        caminho_abs = os.path.abspath(caminho)
        if not os.path.exists(caminho_abs):
            print(f'[Aviso] Arquivo de resultados para {metodo} nao encontrado em {caminho_abs}. Pulando.')
            continue
            
        try:
            df = pd.read_csv(caminho_abs)
            if df.empty:
                continue

            # Detectar coluna de R2 (pode ser 'r2_score' ou 'score_r2')
            r2_col = 'r2_score' if 'r2_score' in df.columns else 'score_r2'
            df[r2_col] = pd.to_numeric(df[r2_col], errors='coerce')
            melhor_indice = df[r2_col].idxmax()
            melhor_linha = df.loc[melhor_indice]
            
            # Identificar colunas de hiperparametros dinamicamente
            cols_parametros = [
                col for col in df.columns 
                if col not in COLUNAS_METRICAS and col not in COLUNAS_BASE
            ]
            
            # Montar a string de parametros
            dict_params = {}
            for col in cols_parametros:
                dict_params[col] = melhor_linha[col]
            
            str_params = ", ".join([f"{k}={v}" for k, v in dict_params.items()]) if dict_params else "N/A"
            
            # Montar dicionario do melhor resultado
            resultado = {
                'metodo': metodo,
                'preprocessamento': melhor_linha.get('preprocessamento', 'N/A'),
                'encoding': melhor_linha.get('encoding', 'N/A'),
                'parametros': str_params,
                'r2_score': melhor_linha.get('r2_score', melhor_linha.get('score_r2', None)),
                'r2_log': melhor_linha.get('r2_log', None),
                'mape': melhor_linha.get('mape', None),
                'mae': melhor_linha.get('mae', None),
                'mse': melhor_linha.get('mse', None),
                'rmse': melhor_linha.get('rmse', None)
            }
            
            # Converter tipos numpy/pandas para tipos python nativos para o JSON
            for k, v in resultado.items():
                if pd.isna(v):
                    resultado[k] = None
                elif hasattr(v, 'item'):  # tipos numpy
                    resultado[k] = v.item()
                    
            melhores.append(resultado)
            
        except Exception as e:
            print(f'[Erro] Falha ao processar {caminho_abs}: {e}')
            
    if not melhores:
        print('[Aviso] Nenhum resultado foi extraido.')
        return
        
    # Salvar em CSV
    df_melhores = pd.DataFrame(melhores)
    df_melhores.to_csv(caminho_csv_saida, index=False)
    print(f'CSV de melhores resultados salvo em: {caminho_csv_saida}')
    
    # Salvar em JSON
    with open(caminho_json_saida, 'w', encoding='utf-8') as f:
        json.dump(melhores, f, indent=4, ensure_ascii=False)
    print(f'JSON de melhores resultados salvo em: {caminho_json_saida}')
    
    # Imprimir no terminal os resultados resumidos
    print('\n' + '=' * 60)
    print('  RESUMO DOS MELHORES RESULTADOS POR METODO')
    print('=' * 60)
    cols_print = [c for c in ['metodo', 'preprocessamento', 'r2_score', 'r2_log', 'mape', 'rmse'] if c in df_melhores.columns]
    print(df_melhores[cols_print].to_string(index=False))
    print('=' * 60 + '\n')

if __name__ == '__main__':
    extrair_melhores()
