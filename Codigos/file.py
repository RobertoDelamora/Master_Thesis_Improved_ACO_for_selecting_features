# -*- coding: utf-8 -*-
"""
Authors: Roberto Alexandre Delamora
E-mail: delamora@gmail.com
Date and hour: 02-02-2023 09:30:00 AM
"""

import pandas as pd
import numpy as np
from math import exp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class FileTreatment(object):
    def __init__(self, database):
        """
        Args:
            self: uma referencia a cada atributo de um objeto criado a partir dessa classe
			database: original database to be analised
            target: target feature identification
        """
        self.file_name = database


    def read_base(self):
        '''
        Faz leitura e carga de base de dados sob análise (base file)
        '''
        self.base = pd.read_csv('data/'+self.file_name, sep=';', decimal=".")
        return self.base
        

    def get_accuracy(self, xData, yData, cl, rep=0):
        """
        Realiza cálculo de acurácia e f1-score utilizando classificadores de ML 
        """
        accuracy = 0
        f1score = 0

        X_train, X_valid, y_train, y_valid = train_test_split(xData, yData, test_size=0.2, random_state=42)
            
        if (cl < 1 or cl > 4):     # Se o valor de cl passado como parametro estiver fora do range, assume valor 1
            cl = 1
        
        if cl == 1:     # KNN Classifier
            # Calculando a acurácia por KNN
            knn = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_valid)

        elif cl == 2:     # MLP Classifier
            rfModel = MLPClassifier(hidden_layer_sizes=1000, random_state=42)
            rfModel.fit(X_train, y_train)
            y_pred = rfModel.predict(X_valid)

        elif cl == 3:     # XGBClassifier
            rfModel = XGBClassifier(n_jobs=-1, eval_metric='mlogloss')
            rfModel.fit(X_train, y_train)
            y_pred = rfModel.predict(X_valid)

        elif cl == 4:     # RandomForest Classifier
            rfModel = RandomForestClassifier(n_jobs=-1, n_estimators=1000, random_state=42)
            rfModel.fit(X_train, y_train)
            y_pred = rfModel.predict(X_valid)

        accuracy = accuracy_score(y_valid, y_pred)
        f1score = f1_score(y_valid, y_pred, average='macro')

        if rep != 0:
            print('Classification Report:')
            print((classification_report(y_valid, y_pred)))

        return f1score, accuracy


    def get_fitness(self, acu, fscore, w1, w2, n_features_total, n_features_track):
        
        '''
        Calcula valor do fitness(G) em função da acurácia do subconjunto de variáveis e da quantidade de variáveis desse subconjunto
        w1 = constante de aderencia à acurácia
        w2 = constante de aderencia à quantidade de variáveis
        n_features_total = total de features do dataset
        n_features_track = número de variáveis da rota da formiga
        
        Retorna: Valor de fitness para a formiga em questão.        
        '''
        # Calcula medida de aprendizado (fitness) em função de acurácia e número de variáveis percorridas pela formiga
        # Adiciona valor de cada formiga no vetor de fitness
        accuracy_weight = w1 * acu
        features_weight = w2 * exp(-n_features_track/n_features_total)

        return (accuracy_weight + fscore + features_weight)

  
    def encoder(self):

        '''
        Faz transformação de variáveis tipo string em números para uso dos classificadores.
        Utiliza método LabelEncoder.
        '''
        
        df = self.base

        # Mapeia variáveis categoricas do dataset
        string_type = ['object']
        categorical_cols = [cname for cname in df.columns if df[cname].dtype in string_type]

        # Substitui valores de variaveis categóricas por números para permitir execução do modelo
        df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)
        
        # Retorna novos valores de xData e yData
        return (df.iloc[:, :-1], df.iloc[:, -1])
  

    def filter_method(self, xData, yData, dist):
        
        '''
        Executa modelo ANOVA para definição dos coeficientes F-scores de cada variável
        Executa função sem fazer seleção de variáveis mas recuperando valor de coeficiente f-score de cada variável.
        Cria pareto para seleção das variáveis
        '''
                
        fvalue_Best = SelectPercentile(f_classif, percentile=100)
        model_fit = fvalue_Best.fit(xData, yData)
        
        # Cria dataframe com informações de scores
        lst_var = xData.columns
        result = pd.DataFrame({'Features':lst_var,
                               'F_Scores':np.round(model_fit.scores_,3),
                               'p-Values':np.round(model_fit.pvalues_,7)}).sort_values(by='F_Scores', ascending=False)

        # Cria colunas de valor acumulado para validação de Pareto
        # Devem permanecer na base somente as veriáveis cujo Acum_% estiver abaixo de 98%
        result['Acum'] = result['F_Scores'].cumsum()
        result['Acum_%'] = result['F_Scores'].cumsum()/result['F_Scores'].sum()*100

        result.reset_index(drop=True, inplace=True)
        
        # Remove variáveis que estão com Acum_% menor que valor definido (95%)
        df1 = result.drop(result[result['Acum_%']>95.00].index, axis=0)
        df1.sort_values(by=['Features'], ascending=True, inplace=True, ignore_index=True)
        features = list(df1['Features'])
        
        # Lista colunas existentes no dataset original
        cols_original = list(xData.columns)
        
        # Identifica indices de colunas removidas
        cols_removed = [cols_original.index(i) for i in cols_original if i not in features]
        
        # Monta novo dataset somente com as variáveis selecionadas
        xData = xData[features]    
        
        # Remove colunas e linhas correspondentes na matriz de distancias
        dist = np.delete(dist, cols_removed, axis=1)
        dist = np.delete(dist, cols_removed, axis=0)

        return (dist, xData)


