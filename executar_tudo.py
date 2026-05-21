# -*- coding: utf-8 -*-
"""
Executar Tudo - Pipeline Completo
Dataset: ds_salaries.csv

Executa em sequência:
  1. Pré-processamento unificado (todas as 4 variações)
  2. Regressão Linear (em todas as bases)
  3. Regressão Polinomial (em todas as bases)

Basta executar este único arquivo para gerar todos os resultados.
"""

import subprocess
import sys
import os

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

python = sys.executable

scripts = [
    ('Pre-processamento Unificado', os.path.join(_dir, 'pre_processamento', 'pre_processamento_unificado.py')),
    ('Regressao Linear',            os.path.join(_dir, 'regressao_linear', 'regressao_linear.py')),
    ('Regressao Polinomial',        os.path.join(_dir, 'regressao_polinomial', 'regressao_polinomial.py')),
]

print('=' * 60)
print('  PIPELINE COMPLETO - Regressao Machine Learning')
print('=' * 60)

for nome, caminho in scripts:
    print(f'\n{"#"*60}')
    print(f'  Executando: {nome}')
    print(f'  Arquivo: {os.path.basename(caminho)}')
    print(f'{"#"*60}\n')

    resultado = subprocess.run(
        [python, caminho],
        cwd=os.path.dirname(caminho),
    )

    if resultado.returncode != 0:
        print(f'\n[ERRO] Falha ao executar {nome}! (codigo: {resultado.returncode})')
        print('Interrompendo o pipeline.')
        sys.exit(1)

print(f'\n{"="*60}')
print('[OK] PIPELINE COMPLETO EXECUTADO COM SUCESSO!')
print(f'{"="*60}')
print(f'\nResultados disponiveis em:')
print(f'  - Regressao Linear:     regressao_linear/resultados_regressao_linear.csv')
print(f'  - Regressao Polinomial: regressao_polinomial/resultados_regressao_polinomial.csv')
print(f'  - Graficos Linear:     regressao_linear/graficos/')
print(f'  - Graficos Polinomial: regressao_polinomial/graficos/')
