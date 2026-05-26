# -*- coding: utf-8 -*-
"""
Análise Exploratória de Dados (EDA)
Dataset: ds_salaries.csv
Target: salary_in_usd

Gera relatório visual com:
  - Distribuição do target
  - Salário por experience_level, company_size, employment_type
  - Top localizações de empresas
  - Contagem de job_group
  - Análise de outliers
  - Correlação entre features numéricas
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

################## Configuração de diretórios ##################

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()

caminho_csv = os.path.join(_dir, '..', 'ds_salaries.csv')
graficos_dir = os.path.join(_dir, '..', 'eda_graficos')
os.makedirs(graficos_dir, exist_ok=True)

################## Leitura e limpeza ##################

base = pd.read_csv(caminho_csv)
if 'Unnamed: 0' in base.columns:
    base = base.drop(columns=['Unnamed: 0'])
base = base.drop(columns=['salary', 'salary_currency'])
base = base.drop_duplicates()

# Feature engineering para EDA
grupos_cargo = {
    'Data Scientist': ['Data Scientist', 'Applied Data Scientist', 'Principal Data Scientist',
                       'Lead Data Scientist', 'Staff Data Scientist', 'Data Science Consultant',
                       'Data Science Engineer', 'Data Specialist'],
    'Data Engineer': ['Data Engineer', 'Big Data Engineer', 'Cloud Data Engineer',
                      'Lead Data Engineer', 'Principal Data Engineer', 'Data Engineering Manager',
                      'Director of Data Engineering', 'Big Data Architect', 'Data Architect',
                      'ETL Developer', 'Analytics Engineer', 'Data Analytics Engineer',
                      'Data Analytics Lead', 'Machine Learning Infrastructure Engineer'],
    'ML Engineer': ['Machine Learning Engineer', 'ML Engineer', 'Lead Machine Learning Engineer',
                    'Machine Learning Scientist', 'Machine Learning Manager',
                    'Machine Learning Developer', 'Head of Machine Learning',
                    'Applied Machine Learning Scientist', 'NLP Engineer',
                    'Computer Vision Engineer', 'Computer Vision Software Engineer',
                    '3D Computer Vision Researcher'],
    'Data Analyst': ['Data Analyst', 'BI Data Analyst', 'Business Data Analyst',
                     'Lead Data Analyst', 'Principal Data Analyst', 'Marketing Data Analyst',
                     'Financial Data Analyst', 'Finance Data Analyst'],
    'Manager/Director': ['Data Science Manager', 'Director of Data Science', 'Head of Data Science',
                         'Head of Data', 'Data Engineering Manager', 'Data Analytics Manager',
                         'Director of Data Engineering'],
    'Research/AI': ['Research Scientist', 'AI Scientist', 'Applied Machine Learning Scientist'],
}

def mapear_cargo(titulo):
    for grupo, titulos in grupos_cargo.items():
        if titulo in titulos:
            return grupo
    return 'Other'

base['job_group'] = base['job_title'].apply(mapear_cargo)
base['is_us_company'] = (base['company_location'] == 'US').astype(int)

# Mapeamento ordinal de experience_level
exp_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
base['exp_label'] = base['experience_level'].map(exp_map)

# Análise de outliers
Q1 = base['salary_in_usd'].quantile(0.25)
Q3 = base['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
base_clean = base[(base['salary_in_usd'] >= lim_inf) & (base['salary_in_usd'] <= lim_sup)]

################## Paleta de cores ##################

CORES = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3']

################## Figura 1: Distribuição do Target ##################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribuição do Target: salary_in_usd', fontsize=14, fontweight='bold')

# Histograma — dados originais
axes[0].hist(base['salary_in_usd'], bins=40, color=CORES[0], edgecolor='white', alpha=0.85)
axes[0].set_title(f'Original (com outliers)\nN={len(base)} | Mediana: ${base["salary_in_usd"].median():,.0f}')
axes[0].set_xlabel('Salário (USD)')
axes[0].set_ylabel('Frequência')
axes[0].axvline(base['salary_in_usd'].median(), color='red', linestyle='--', label=f'Mediana')
axes[0].axvline(base['salary_in_usd'].mean(), color='orange', linestyle='--', label=f'Média')
axes[0].legend()

# Histograma — após remoção de outliers
axes[1].hist(base_clean['salary_in_usd'], bins=40, color=CORES[2], edgecolor='white', alpha=0.85)
axes[1].set_title(f'Após remoção de outliers IQR\nN={len(base_clean)} | Mediana: ${base_clean["salary_in_usd"].median():,.0f}')
axes[1].set_xlabel('Salário (USD)')
axes[1].set_ylabel('Frequência')
axes[1].axvline(base_clean['salary_in_usd'].median(), color='red', linestyle='--', label='Mediana')
axes[1].axvline(base_clean['salary_in_usd'].mean(), color='orange', linestyle='--', label='Média')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '01_distribuicao_target.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 01_distribuicao_target.png')

################## Figura 2: Log-Transform do Target ##################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Efeito do Log-Transform em salary_in_usd', fontsize=14, fontweight='bold')

axes[0].hist(base['salary_in_usd'], bins=40, color=CORES[0], edgecolor='white', alpha=0.85)
axes[0].set_title('Escala Original (assimétrica)')
axes[0].set_xlabel('salary_in_usd')

axes[1].hist(np.log1p(base['salary_in_usd']), bins=40, color=CORES[4], edgecolor='white', alpha=0.85)
axes[1].set_title('Log1p Transform (mais simétrica)')
axes[1].set_xlabel('log1p(salary_in_usd)')

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '02_log_transform.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 02_log_transform.png')

################## Figura 3: Salário por Experience Level ##################

exp_order = ['Entry', 'Mid', 'Senior', 'Executive']
exp_data = [base_clean[base_clean['exp_label'] == e]['salary_in_usd'].values for e in exp_order]
exp_medians = [np.median(d) for d in exp_data]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Salário por Nível de Experiência (sem outliers)', fontsize=14, fontweight='bold')

bp = axes[0].boxplot(exp_data, labels=exp_order, patch_artist=True, notch=False)
for patch, cor in zip(bp['boxes'], CORES):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
axes[0].set_ylabel('Salário (USD)')
axes[0].set_title('Boxplot por Experience Level')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

axes[1].bar(exp_order, exp_medians, color=CORES[:4], alpha=0.85, edgecolor='white')
axes[1].set_ylabel('Mediana do Salário (USD)')
axes[1].set_title('Mediana do Salário por Experience Level')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
for i, v in enumerate(exp_medians):
    axes[1].text(i, v + 1000, f'${v/1000:.0f}K', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '03_salario_por_experience.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 03_salario_por_experience.png')

################## Figura 4: Salário por Grupo de Cargo ##################

job_stats = base_clean.groupby('job_group')['salary_in_usd'].agg(['median', 'count']).sort_values('median', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(job_stats.index, job_stats['median'], color=CORES[:len(job_stats)], alpha=0.85, edgecolor='white')
ax.set_xlabel('Mediana do Salário (USD)')
ax.set_title('Mediana do Salário por Grupo de Cargo (sem outliers)', fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
for bar, (_, row) in zip(bars, job_stats.iterrows()):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f'${row["median"]/1000:.0f}K (n={int(row["count"])})', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '04_salario_por_cargo.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 04_salario_por_cargo.png')

################## Figura 5: EUA vs Resto do Mundo ##################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Impacto de Empresa nos EUA vs Outros Países', fontsize=14, fontweight='bold')

us_data   = base_clean[base_clean['is_us_company'] == 1]['salary_in_usd']
nao_us    = base_clean[base_clean['is_us_company'] == 0]['salary_in_usd']

bp = axes[0].boxplot([us_data.values, nao_us.values],
                     labels=['EUA', 'Outros'], patch_artist=True)
bp['boxes'][0].set_facecolor(CORES[0]); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(CORES[1]); bp['boxes'][1].set_alpha(0.7)
axes[0].set_ylabel('Salário (USD)')
axes[0].set_title('Boxplot: EUA vs Outros')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

axes[1].bar(['EUA', 'Outros'], [us_data.median(), nao_us.median()],
            color=[CORES[0], CORES[1]], alpha=0.85, edgecolor='white')
axes[1].set_ylabel('Mediana do Salário (USD)')
axes[1].set_title(f'EUA: ${us_data.median()/1000:.0f}K | Outros: ${nao_us.median()/1000:.0f}K')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
for i, v in enumerate([us_data.median(), nao_us.median()]):
    axes[1].text(i, v + 500, f'${v/1000:.0f}K', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '05_us_vs_outros.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 05_us_vs_outros.png')

################## Figura 6: Salário por Tamanho de Empresa ##################

size_order = ['S', 'M', 'L']
size_labels = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
size_data = [base_clean[base_clean['company_size'] == s]['salary_in_usd'].values for s in size_order]
size_medians = [np.median(d) if len(d) > 0 else 0 for d in size_data]
size_counts  = [len(d) for d in size_data]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([size_labels[s] for s in size_order], size_medians,
              color=CORES[:3], alpha=0.85, edgecolor='white')
ax.set_ylabel('Mediana do Salário (USD)')
ax.set_title('Mediana do Salário por Tamanho de Empresa (sem outliers)', fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
for bar, med, cnt in zip(bars, size_medians, size_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'${med/1000:.0f}K\n(n={cnt})', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '06_salario_por_company_size.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 06_salario_por_company_size.png')

################## Figura 7: Análise de Outliers ##################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Análise de Outliers — salary_in_usd (IQR)\nLimite inferior: ${lim_inf:,.0f} | Limite superior: ${lim_sup:,.0f}',
             fontsize=13, fontweight='bold')

axes[0].boxplot(base['salary_in_usd'].values, patch_artist=True,
                boxprops=dict(facecolor=CORES[0], alpha=0.7))
axes[0].set_title(f'Com outliers (N={len(base)})')
axes[0].set_ylabel('Salário (USD)')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

axes[1].boxplot(base_clean['salary_in_usd'].values, patch_artist=True,
                boxprops=dict(facecolor=CORES[2], alpha=0.7))
axes[1].set_title(f'Sem outliers (N={len(base_clean)})')
axes[1].set_ylabel('Salário (USD)')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '07_analise_outliers.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 07_analise_outliers.png')

################## Figura 8: Correlação entre features numéricas ##################

# Colunas numéricas disponíveis
cols_num = ['work_year', 'remote_ratio', 'salary_in_usd', 'is_us_company']
corr = base[cols_num].corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_num)))
ax.set_yticks(range(len(cols_num)))
ax.set_xticklabels(cols_num, rotation=30, ha='right')
ax.set_yticklabels(cols_num)
plt.colorbar(im, ax=ax, label='Correlação de Pearson')
for i in range(len(cols_num)):
    for j in range(len(cols_num)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                fontsize=10, color='black' if abs(corr.values[i, j]) < 0.6 else 'white')
ax.set_title('Matriz de Correlação — Features Numéricas', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, '08_correlacao_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] 08_correlacao_features.png')

################## Relatório de Estatísticas ##################

print('\n' + '='*65)
print('RELATÓRIO DE ESTATÍSTICAS — ds_salaries.csv')
print('='*65)
print(f'Total de registros (sem duplicatas): {len(base)}')
print(f'Registros após remoção de outliers:  {len(base_clean)}')
print(f'\nsalary_in_usd:')
print(f'  Mínimo:      ${base["salary_in_usd"].min():>10,.0f}')
print(f'  Q1:          ${Q1:>10,.0f}')
print(f'  Mediana:     ${base["salary_in_usd"].median():>10,.0f}')
print(f'  Média:       ${base["salary_in_usd"].mean():>10,.0f}')
print(f'  Q3:          ${Q3:>10,.0f}')
print(f'  Máximo:      ${base["salary_in_usd"].max():>10,.0f}')
print(f'  Outliers:    {len(base) - len(base_clean)} ({100*(len(base)-len(base_clean))/len(base):.1f}%)')
print(f'\nDistribuição por experience_level:')
for exp, label in [('EN','Entry'),('MI','Mid'),('SE','Senior'),('EX','Executive')]:
    cnt = (base['experience_level'] == exp).sum()
    pct = 100 * cnt / len(base)
    med = base[base['experience_level'] == exp]['salary_in_usd'].median()
    print(f'  {label:12s}: {cnt:4d} ({pct:5.1f}%) | Mediana: ${med:>8,.0f}')
print(f'\nDistribuição por job_group:')
for grp, cnt in base['job_group'].value_counts().items():
    med = base[base['job_group'] == grp]['salary_in_usd'].median()
    print(f'  {grp:20s}: {cnt:4d} | Mediana: ${med:>8,.0f}')
print(f'\nEmpresa nos EUA: {base["is_us_company"].sum()} ({100*base["is_us_company"].mean():.1f}%)')
print(f'  Mediana EUA: ${us_data.median():,.0f}')
print(f'  Mediana Outros: ${nao_us.median():,.0f}')
print(f'\n[OK] Gráficos salvos em: {os.path.abspath(graficos_dir)}')
print('='*65)
