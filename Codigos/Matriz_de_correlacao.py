# -*- coding: utf-8 -*-
"""
Authors: Roberto Alexandre Delamora
E-mail: delamora@gmail.com
Date and hour: 02-02-2023 09:30:00 AM

Este código é utilizado para criação da matriz de correlação das bases sob estudo
"""


import pandas as pd
import numpy as np
import phik

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Estebelece limites para visualização no notebook
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',500)

# Limita a 3 casas decimais a apresentação das variaveis tipo float
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x)) 



# ========================================
# Funções recorrentes

# Remove colunas com alto indice de campos sem informação ou campos com informação repetida
# Devem ser indicados o dataframe para análise e o limite aceitável (percentual)
# Retorna com lista de colunas para remoção

def get_useless_cols(df, perc, param=3):
    
    null_cols = []
    rep_cols  = []

    if (param == '') | (param == 1):      # Não foi passado o parametro ou param é 1. Considera somente valores nulos
        null_cols = [col for col in df.columns if (df[col].isnull().sum() / df.shape[0]) >= perc]
        print('Colunas com mais de {0:5.1f}% de campos nulos     : {1:3d}'.format(perc*100, len(null_cols)))

    elif param == 2:                   # Considera somente dados repetidos
        rep_cols = [col for col in df.columns if (df[col].value_counts(dropna=False, normalize=True).values[0]) >= perc]
        print('Colunas com mais de {0:5.1f}% de campos repetidos : {1:3d}'.format(perc*100, len(rep_cols)))

    elif param == 3:                   # Considera valores nulos e repetidos
        null_cols = [col for col in df.columns if (df[col].isnull().sum() / df.shape[0]) >= perc]
        rep_cols = [col for col in df.columns if (df[col].value_counts(dropna=False, normalize=True).values[0]) >= perc]
        print('Colunas com mais de {0:5.1f}% de campos nulos.....: {1:3d}'.format(perc*100, len(null_cols)))
        print('Colunas com mais de {0:5.1f}% de campos repetidos.: {1:3d}'.format(perc*100, len(rep_cols)))
    
    cols_to_drop = sorted(list(set(null_cols + rep_cols)))
    
    return cols_to_drop

# ========================================


# Tratamento básico dos arquivos
# Seleciona uma base de cada vez
dset = 'wine.data'
#dset = 'soybean-small.data'
#dset = 'ionosphere.data'
#dset = 'breast-cancer.data'
#dset = 'hill_valley_without_noise_training.data'
#dset = 'arrhythmia.data'
#dset = 'madelon_train.data'

base_name = dset[:-5]
base_name

# Le arquivo csv e transforma em dataframe
if (base_name[:4] == 'hill'):
    xData = pd.read_csv('bases/'+dset, sep=',')
elif base_name == 'madelon_train':
    xData = pd.read_csv('bases/'+dset, header=None, sep=' ')
    yData = pd.read_csv('bases/madelon_train.labels', header=None, sep=',')
    xData[501] = yData
else:    
    xData = pd.read_csv('bases/'+dset, header=None, sep=',')


for i in xData.columns:
    xData[i] = xData[i].replace('?', np.nan)
    xData[i] = xData[i].fillna(xData[i].mode()[0])

# Coloca variável target como último atributo da tabela
if base_name =='wine':
    nova_ordem = xData.columns[1:14]
    nova_ordem = nova_ordem.append(xData.columns[:1])
    xData = xData[nova_ordem]
    xData.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
elif base_name == 'breast-cancer':
    xData.drop([0], axis=1, inplace=True)
    xData.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Mapeia tipos de cada uma das variáveis
int_cols = [cname for cname in xData.columns if xData[cname].dtype in ['int8', 'int16', 'int32', 'int64']]
flt_cols = [cname for cname in xData.columns if xData[cname].dtype in ['float16', 'float32', 'float64']]
cat_cols = [cname for cname in xData.columns if xData[cname].dtype == 'object']
dat_cols = [cname for cname in xData.columns if xData[cname].dtype == 'datetime64[ns]']

# Altera nomes das variáveis dos datasets para facilitar identificação em caso de remoção
# Aplicar somente para datasets que não possuem nomes específicos para as variáveis
if base_name in(['wine', 'soybean-small', 'ionosphere', 'breast-cancer', \
                 'hill_valley_without_noise_training', 'arrhythmia', 'madelon_train']):
    lst_col = []
    for i in xData.columns:
        lst_col.append('v_'+str(i))
    
    xData.columns = lst_col

# Cria novos dataframes com variáveis independentes e target
yData = xData.iloc[:, -1]
xData = xData.iloc[:, :-1]

# Remove colunas com indice de variáveis repetidas ou nulas acima do indicado
# Passado parametro 1, então função considera somente valores nulos
# Se quantidade de nulos for maior que o percentual indicado, a coluna é removida
to_drop_cols = get_useless_cols(xData, 0.9999, 3)
print('Colunas para remoção : ', to_drop_cols)

xData.drop(to_drop_cols, axis=1, inplace=True)
print('\nDataframe resultante após remoção das colunas indicadas')
xData.head(2)

# Seleciona variáveis com baixa cardinalidade (abaixo de 2)
# É arbritário mas remove variáveis com pouco conteúdo informacional
num_samples = len(xData)
low_cardinality_cols = [cname for cname in xData.columns if xData[cname].nunique() == 1]
print('Colunas a serem removidas pois contém somente um único valor de informação:\n', low_cardinality_cols)
print('\nDataframe resultante após remoção das colunas indicadas')

# Remove variáveis com baixa cardinalidade listadas anteriormente
xData.drop(low_cardinality_cols, axis=1, inplace=True)
xData.head(2)

# Retornando a variável target ao dataset
xData = pd.concat([xData, yData], axis=1)

# Grava arquivo de base que será utilizada no algoritmo
xData.to_csv('data/'+base_name+'.csv', sep=';', encoding='utf-8', index=False)
print('Base name = ', base_name)
print('Arquivo gerado = ', base_name+'.csv')

# Remove novamente variável target em preparação à montagem da matriz de correlações
xData = xData.iloc[:, :-1]

# Cria matriz de correlação considerando todas as variáveis
# Cálculo de correlação phik
# Monta matriz de correlações que será matriz de distancias
distancia = xData.phik_matrix()   # Monta matriz de correlações que será matriz de distancias
distancia = pd.DataFrame(data=distancia)

# Substitui valores NAN na matriz por 0
distancia.replace(np.nan, 0, inplace=True)

# Preenche diagonal com valores 1
np.fill_diagonal(distancia.values, 1)

# Obtém valor absoluto de todos os dados
distancia = abs(distancia)

# Calcula valores minimo e máximo dentro da matriz
min = distancia.values.min()
max = distancia.values.max()
dif = distancia.values.max()-distancia.values.min()

print(min)
print(max)
print(dif)

# Redimensiona valores da matriz para que fiquem entre 1 e 10
distancia = (9*(distancia - min)/dif)+1
distancia = pd.DataFrame(data=distancia)
distancia

# Grava arquivo de base que será utilizada no algoritmo
distancia.to_csv('data/'+base_name+'_distances.csv', sep=';', encoding='utf-8', index=False)
print('Base name = ', base_name)
print('Arquivo gerado = ', base_name+'_distances.csv')

# Ao final, deverão existir 2 novos arquivos na pasta data:
# 1) A base de dados tratada e pronta para uso pelo algoritmo ACO
# 2) A matriz de correlação
