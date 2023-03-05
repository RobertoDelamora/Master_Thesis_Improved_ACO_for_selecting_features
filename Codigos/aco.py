# -*- coding: utf-8 -*-
"""
Authors: Roberto Alexandre Delamora
E-mail: delamora@gmail.com
Date and hour: 02-02-2023 09:30:00 AM
"""


import argparse
import pandas as pd
import time
import gc
from antColony import AntColony

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACO')
    parser.add_argument('-p', '--ro', type=float, dest='p', default='0.15', help='pheromone rate decay')
    parser.add_argument('-a', '--alpha', type=float, dest='a', default='2.0', help='pheromone trail importance')
    parser.add_argument('-b', '--beta', type=float, dest='b', default='1.0', help='local heuristic importance')
    parser.add_argument('-fi', '--fi_constant', type=float, dest='phi', default='0.50', help='fi on pheromone upgrade equation')
    parser.add_argument('-i', '--iterations', type=int, dest='i', default='10', help='iterations number')
    parser.add_argument('-f', '--data_base', dest='data_base', default='wine.csv', help='database file')
    parser.add_argument('-w1', '--w1_constant', type=int, dest='w1', default='150', help='w1 on fitness equation')
    parser.add_argument('-w2', '--w2_constant', type=int, dest='w2', default='2', help='w2 on fitness equation')
    parser.add_argument('-cl', '--classifier', type=int, dest='cl', default='2', help='type of classifier used to calculate accuracy')
    parser.add_argument('-rd', '--rounds', type=int, dest='rounds', default='5', help='number of rounds executed by algorithm')

    args = parser.parse_args()

    # Define classificadores usados no processo de cálculo da acurácia (parte Wrapper do algoritmo)
    classifier = {1: 'KNN', 2: 'MLP', 3: 'XGBoost', 4: 'Random Forest'}

    bases = ['wine', 'soybean-small', 'ionosphere', 'breast-cancer', 'hill_valley_without_noise_training', 'arrhythmia', 'madelon_train']
    itera = [10, 20, 30]
    num_class = [1, 2, 3, 4]
    loops = 3


    # Cria dataframe com informação sobre lista de variaveis e acurácia
    var = pd.DataFrame(columns=['iterações', 'base', 'classifier', 'tempo', 'acuracia', 'f1score', 'features', 'media_acur', 'std_acur', 'media_fscore', 'std_fscore'])   
    var.loc[0, 'iterações'] = 0
    var.loc[0, 'base'] = ''
    var.loc[0, 'classifier'] = ''
    var.loc[0, 'tempo'] = 0
    var.loc[0, 'acuracia'] = 0
    var.loc[0, 'features'] = ''   
    var.loc[0, 'media_acur'] = 0
    var.loc[0, 'std_acur'] = 0

    # Cria arquivo xlsx de modelo para armazenamento dos resultados apurados durante a seleção    
    var.to_excel('data/medidas.xlsx', float_format="%8.4f", sheet_name='Medidas', index=False)

    # Inicializa variáveis
    acuracia = 0
    f1_score = 0
    indice = 0
    
    # Executa ACO no número de loops definido e com parâmetros estabelecidos
    for num_iter in itera:
    	for base in bases:
            for key in num_class:
                for i in range(0, loops):

                    # Faz leitura das bases de dados
                    # Os arquivos dos datasets devem estar no formato .csv com dados separados por ponto-e-vírgula
                    # localizados numa pasta "data/" e o nome deve seguir padrão "nome da base+_distances".
                    matrix = base + '_distances.csv'
                    distance_matrix = pd.read_csv('data/'+matrix, header=0, sep=';').values
            	
                    # Cria vetor com valores de rho, alpha, beta, phi, w1 e w2
                    v_param = [args.p, args.a, args.b, args.phi, args.w1, args.w2]
            
                    # Inicia algoritmo com classificador selecionado no parametro de entrada
                    print()
                    #print('\nStarting ACOFS_rank running over dataset "{}" with "{}" classifier.\n'.format(args.data_base, classifier.get(args.cl))) 
                    print('\nStarting ACOFS_rank running over dataset "{}" with "{}" classifier.\n'.format(base, classifier.get(key)))
                    print()
                    print('Ciclo -> ', i+1, ' do Classificador: ', classifier.get(key))
            
                
                    '''
                    Instancia classe AntColony no objeto colony
                    ===========================================
                    Ponto importante: nvar_data e a lista com números mínimo e máximo de variáveis que cada formiga poderá percorrer.
                    nvar_data também representa o range de variáveis que se deseja selecionar na pesquisa pela melhor opção.
                    As formigas irão percorrer uma quantidade de nós que será definida aleatoriamente para cada uma
                    '''				   
                    colony = AntColony(distance_matrix, num_iter, v_param, key, args.rounds, i+1, classifier.get(key))
            
                    time_ini = time.time()
                
                    # Chama método run (Function to run the rank based Ant Colony Optimization algorithm) do objeto colony e armazena em best_ant
                    best_ant = colony.run(base+'.csv')
            
                    time_fim = time.time()
                    time_exec = time_fim - time_ini
                    print('Running time (sec.)  : {:3.2f}'.format(time_exec))
                    print()
                    var.loc[indice] = [num_iter,                   # Numero de iterações
                                       base,                       # Base de dados sob análise
                                       classifier.get(key),        # Tipo de classificador escolhido
                                       round(time_exec, 2),        # Tempo total de execução
                                       best_ant[0],                # Acurácia da melhor solução
                                       best_ant[1],                # f1-score da melhor solução
                                       best_ant[2],                # Número de variáveis da melhor solução
                                       round(best_ant[3], 4),      # Valor médio de acurácia entre as melhores soluções apresentadas
                                       round(best_ant[4], 4),      # Desvio padrão entre as melhores soluções apresentadas
                                       round(best_ant[5], 4),      # Valor médio de f1-score entre as melhores soluções apresentadas
                                       round(best_ant[6], 4)]      # Desvio padrão entre as melhores soluções apresentadas
                    
                    # Grava arquivo de base que será utilizada como registro das medidas
                    #var.to_csv('data/'+base+'_medidas.csv', sep=';', encoding='utf-8', index=False)
                    var.to_excel('data/medidas.xlsx', float_format="%8.4f", sheet_name='Medidas', index=False)

                    indice+=1

    best_ant.insert(0, args.data_base)
    best_ant.insert(1, 'classifier')
    best_ant.insert(2, round(time_exec, 2))
    
    del colony, distance_matrix, matrix
    del time_exec, time_fim, time_ini
    del args, parser, classifier, v_param
    del acuracia, base, bases, best_ant, i, f1_score
    del indice, itera, key, num_class, num_iter, var
    gc.collect()    
    					   