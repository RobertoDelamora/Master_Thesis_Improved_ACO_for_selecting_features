# -*- coding: utf-8 -*-

from ant import Ant
import numpy as np
import pandas as pd
import random
from file import FileTreatment
from random import randrange

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class AntColony(object):
    def __init__(self, distances, n_iterations, param, cl, rounds, cicle, classificador):
        """
        Args:
            self: uma referencia a cada atributo de um objeto criado a partir dessa classe
			distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            distance_cost: Cost of one unit of distance
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            rho (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=2
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
            num_atributes (int): number of features to be selected as the best.
            q = constant 
            w1 = parameter of weight of accuracy in fitness
            w2 = parameter of weight of number of features in fitness
            phi = parameter of weight of accuracy in pheromone upgrade
            cl = number of classifier used to calculate accuracy
            classificador = name of classifier used
        Example:
            ant_colony = AntColony(german_distances, 10, [rho, alpha, beta, phi, w1, w2] , 1, 3, 2, 'KNN')          
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape)   # Cria matriz com mesmo tamanho da matriz de distancias e preenchida com 1
        self.n_iterations = n_iterations
        self.rho = param[0]
        self.alpha = param[1]
        self.beta = param[2]
        self.phi = param[3]
        self.w1 = param[4]
        self.w2 = param[5]
        self.cl = cl
        self.rounds = rounds
        self.features = []
        self.cicle = cicle
        self.classificador = classificador


    def run(self, database):
        '''
        Function to run the rank based Ant Colony Optimization algorithm
        '''

        '''
        Cria o objeto file como instância da classe FileTreatment passando parametros data_base (nome do arquivo de dados) e target_feature
        No construtor de FileTreatment define parametros:
            - file_name = data_base
            - target = target_feature
        '''
        file = FileTreatment(database)

        # Identifica qual é o arquivo da base de dados
        file.read_base()

        #xData, yData = file.encoder(xData, yData)
        
        # Valida variáveis categóricas e transforma em numéricas através da função Encoder()
        # Faz split do dataset separando variáveis independentes de target
        xData, yData = file.encoder()

        # Calcula f1_score e accuracy considerando todas as variáveis da base original 
        # Nesse ponto, todas as variáveis do dataset já são numéricas
        f1_score, accuracy = file.get_accuracy(xData.values, yData.values, self.cl)
        
        # Cria dataframe com informação sobre lista de variaveis e acurácia
        lst_var = pd.DataFrame(columns=['round', 'num_features', 'features', 'accuracy', 'f1_score'])   
        lst_var.loc[0, 'round'] = 1
        lst_var.loc[0, 'num_features'] = len(xData.columns)   # Adiciona linha com numero de variáveis definido
        lst_var.loc[0, 'features'] = 'All'   
        lst_var.loc[0, 'accuracy'] = round(accuracy, 4)
        lst_var.loc[0, 'f1_score'] = round(f1_score, 4)

        print('Accuracy with all {:3d} features = {:.4f}'.format(len(xData.columns), accuracy))
        
        '''
        Executa primeira etapa de seleção de variáveis pelo Método Filter
        Esta etapa manterá somente certo número de variáveis que atende aos requisitos
        Retorna arquivo somente com variáveis restantes que deverão ser consideradas na segunda etapa 
        que utiliza Método Wrapper (ACO)
        '''    
        # Aplica seleção de variáveis pelo método Filter
        # Faz seleção de variáveis pelo teste ANOVA e seleciona o número de variáveis
        # que responde a 85% da explicabilidade do comportamento da variável target        
        self.distances, xData = file.filter_method(xData, yData, self.distances)

      	# Define o número de n_best_ants como proporção (30%) da quantidade de features do dataset, limitado a 15 formigas elitistas
        n_best_ants = int(round((self.distances.shape[0]*0.30), 0))
        self.n_best = min(15, n_best_ants)

        # Número total de formigas do algoritmo é igual ao número de variáveis do dataset        
        self.n_ants = self.distances.shape[0]   

        '''
        Número mínimo de variaveis a serem percorridas é 3
        Número máximo de variaveis a serem percorridas é o numero total de variáveis
        '''
        min_nvar = 3   
        max_nvar = self.distances.shape[0]
        self.n_var = [min_nvar, max_nvar]
    
        print('\nMinimo de variáveis  : ', min_nvar)
        print('Máximo de variáveis  : ', max_nvar)
        print()


        # Inicia rodadas ou experimentos
        for rodada in range(self.rounds):

            print('\nCicle: ', self.cicle, ' - Round: ',rodada+2, ' - Classifier: ', self.classificador)
            print('==========')

            # Cria matriz base com mesmo tamanho da matriz de distancias e com valores iniciais de feromonio preenchidos com 1
            self.pheromone = np.ones(self.distances.shape)   

            '''
            Cria o objeto best_ant como instância da classe Ant passando parametros start_node = 0 e size = 0
            No construtor de ant define parametros:
                - path = [start_node]
                - of_value = infinito
                - distance = 0.0
                - possible_nodes = [x for x in range(size) if x != start_node]
                '''
            best_ant = Ant(0, 0)   


            # Loop de iterações (épocas) com construção das soluções e avaliação da melhor opção
            # Executa loop pelo número de vezes definido em n_iterations que por padrão é 1
            for i in range(self.n_iterations):
                colony = []
   			
                # Define variavel de controle da posição inicial de cada formiga
                x = []  

                # Define lista de nós possíveis de serem percorridos
                possible_nodes = list(range(0, self.distances.shape[0]))

                # Escolhe aleatoriamente posição inicial de uma formiga
                start_ant = np.random.randint(self.distances.shape[0])   

                # Executa loop tantas vezes quanto o número de formigas (n_ants)
                for j in range(self.n_ants):   

                    # Processo para garantir que cada formiga irá partir de uma variável e que a escolha é aleatória
                    while start_ant in x:
                        start_ant = random.choice(possible_nodes)
                    x.append(start_ant)    
                    possible_nodes.remove(start_ant)

                    '''
                    Cria o objeto ant como instância da classe Ant passando parametros:
                        - start_node = valor aleatório entre  0 e o número de linhas da matriz distancies
                        - size = total de linhas da matriz distances
                    Retorna: Lista de possíveis nós que a formiga pode percorres (todos menos o nó em que ela está que é definido aleatoriamente - x)
                    '''
                    ant = Ant(start_ant, self.distances.shape[0])

                    n_var_track = randrange(self.n_var[0], self.n_var[1]+1)    # Define aleatoriamente o número de variáveis do caminho dessa formiga    

                    '''
                    Executa método build_path (build a path for an ant) do objeto ant
                    Passando como parametros distances, pheromone, alpha, beta e a lista n_var
                    Retorna: Caminha a ser percorrido por uma formiga
                    '''
                    path = ant.build_path(self.distances, self.pheromone, self.alpha, self.beta, n_var_track, possible_nodes)
    
                    '''
                    Compara caminho escolhido com caminhos existentes para remover rotas com variáveis identicas
                    Se o novo caminho for igual a um já existente o modelo tentará gerar um novo caminho 5 vezes
                    Se não for gerado um novo caminho, será então aceita a duplicidade
                    Esse limite é para evitar que o algoritmo entre em loop já que a definição do caminho é um processo aleatório
                    '''
                    path = ant.compare_path(path, colony, ant, self.n_var, self.distances, self.pheromone, self.alpha, self.beta, n_var_track, possible_nodes)
                    
                    '''
                    Executa método calculate_distance (calculate the distance traveled by the ant) do objeto ant
                    Passando parametro distances
                    Retorna: distancia percorrida por uma formiga (sum distances[current_node, self.path[0]])
                    '''
                    ant.calculate_distance(self.distances)
                    
                    xMat = xData.iloc[:,path].values   # Valores das nvar variáveis selecionadas
                    yMat = yData.values
                    f1_score, accuracy = file.get_accuracy(xMat, yMat, self.cl)
                    ant.of_value = accuracy   # Calcula acurácia do trecho percorrido
                    ant.f1score = f1_score    # Calcula f1_score do trecho percorrido

                    # Calcula medida de aptidão (fitness) em função de acurácia e número de variáveis percorridas pela formiga
                    # Adiciona valor de cada formiga no vetor de fitness
                    ant.fitness = file.get_fitness(accuracy, f1_score, self.w1, self.w2, self.distances.shape[0], len(path))
                    
                    '''
                    Adiciona registro à lista colony com parametros do objeto ant:
                        - distance
                        - of_value
                        - path
                        - possible_nodes
                    Ao final do loop colony terá n_ants registros com os valores dos parametros de cada formiga
                    '''
                    colony.append(ant)
                

                '''
                Executa método n_bests_ants (Function to find the n ants that will update the pheromone trail)
                Passando lista colony como parametro
                A função retorna lista colony ordenada pelo parametro fitness em ordem decrecente
                A melhor formiga está no topo da lista (posição 0)
                '''
                lst_best_ants = self.find_n_bests_ants(colony)
                
    			
                '''
                Verifica se o melhor valor da lista de melhores formigas é maior que o valor atual armazenado da melhor formiga
                Se verdadeiro, aualiza objeto best_ant com parametros da melhor formiga de colony
                '''
                if lst_best_ants[0].fitness > best_ant.fitness:    # Maximizando resultado
                    best_ant = lst_best_ants[0]
                    accur = lst_best_ants[0].of_value
                    f1 = lst_best_ants[0].f1score
                    lst_feat = lst_best_ants[0].path
                    print("Iteraction {:2d} - Best OF  -> accuracy = {:.4f} with {:3d} features".format(i+1, accur, len(lst_feat)))
                else:
                    print("Iteraction {:2d} - No gains with {:3d} features".format(i+1, len(lst_best_ants[0].path)))


                self.update_pheromone(best_ant, lst_best_ants)
            # loop de iterações termina aqui

            lst_feat.sort()
            lst_var.loc[rodada+1] = [rodada+2, len(lst_feat), lst_feat, round(accur, 4), round(f1, 4)]
                        
            print()        
            print(lst_var)
            print()
        # loop de rodadas termina aqui


        mean_acc = lst_var['accuracy'].mean()
        std_acc = lst_var['accuracy'].std()
        mean_fscore = lst_var['f1_score'].mean()
        std_fscore = lst_var['f1_score'].std()
        
  
        total_iterations = (i+1) * (rodada+1) + 1   # Numero total de iterações
        # Processo da seleção da melhor solução: maior acurácia com menor número de variáveis
        best_choice = lst_var.query('accuracy == accuracy.max()')
        best_choice = best_choice.query('num_features == num_features.min()')
        best_choice = best_choice.query('f1_score == f1_score.max()')

        
        # Converte listas em string para poder usar função drop_duplicates()
        best_choice[['features']] = best_choice[['features']].astype(str)
        best_choice['features'] = best_choice['features'].str.strip('[]')
        
        # Removendo linhas com listas duplicadas do dataframe de resultados
        best_choice.drop_duplicates(subset=['features'], keep='first', inplace = True)

        best_choice.reset_index(drop=True, inplace=True)
        
        # Se a melhor opção não for aquela com o máximo de variáveis
        if best_choice.loc[0, 'features'] != 'All':
            # Separa variáveis novamente e transforma em lista
            best_choice['features'] = best_choice['features'].str.split(', ')

            # Identifica variáveis no dataset original
            for row in range(0, len(best_choice)):
                lista = best_choice.loc[row, 'features']
                lista = [eval(i) for i in lista]
                best_choice.at[row, 'features'] = lista

        
        # Calcula Fator de redução
        fator_redu = (len(self.distances) - best_choice['num_features'][0])/len(self.distances)*100
        classifier = {1: 'KNN', 2: 'MLP', 3: 'XGBoost', 4: 'Random Forest'}
        
        # Se não houver ganho com redução de variáveis
        if fator_redu <= 0:
            print('\nFinal result')
            print('============')        
            print('Using {} classifier'.format(classifier.get(self.cl)))
            print("The model didn't find gains with feature reduction comparing with accuracy for all features")
            print('Best OF after {:2d} iterations in {:2d} rounds was {:.4f} with {:3d} features'.format(total_iterations, rodada+2, best_choice['accuracy'][0], best_choice['num_features'][0]))
            print('Accuracy Average ............:  {:.3f}'.format(mean_acc))
            print('Accuracy Standard Deviation .:  {:.3f}'.format(std_acc))
            print()
            
        else:
            print('\nFinal result')
            print('============')        
            print('Using {} classifier'.format(classifier.get(self.cl)))
            print('Best OF after {:2d} iterations in {:2d} rounds was {:.4f} with {:3d} features'.format(total_iterations, rodada+2, best_choice['accuracy'][0], best_choice['num_features'][0]))
            print('Features reduction factor ...: {:.2f} %'.format(fator_redu))
            print('Accuracy Average ............:  {:.3f}'.format(mean_acc))
            print('Accuracy Standard Deviation .:  {:.3f}'.format(std_acc))
            print()
            print('List of possible solutions :\n')
            for z in range(len(best_choice)):
                xMat = xData.iloc[:,best_choice.loc[z, 'features']]
                list_features = xMat.columns
                print(list_features.tolist())
                xMat = xData.loc[:,list_features].values   # Valores das nvar variáveis selecionadas
                yMat = yData.values
                f1_score, accuracy = file.get_accuracy(xMat, yMat, self.cl, 1)
                print('\n')           

        return [best_choice['accuracy'][0], best_choice['f1_score'][0], best_choice['num_features'][0], round(mean_acc, 4), round(std_acc, 4), round(mean_fscore, 4), round(std_fscore, 4)]
    


    def find_n_bests_ants(self, colony):
        '''
        Function to find the n ants that will update the pheromone trail
        '''
        '''
        Executa método padrão sort do objeto colony considerando método of (Function to return the of value) do objeto ant
        A lista colony é ordenada pelo parametro of_value de forma crescente
        Retorna: lista colony ordenada
        '''
        colony.sort(key=lambda ant: ant.of(), reverse=True)   # Sort deve ser por ordem decrescente - Maximizando resultados
        
        return colony[:self.n_best]
    
    
    def go_through_path(self, ant, factor):
        '''
        Function to calculate the amount of pheromone to be deposited in trail piece 
        '''
        current_node = ant.path[0]

        '''
        Executa loop por n_ants vezes (0 até n_ants-1)
        Nó inicial é o da melhor formiga
        Nó seguinte é o da segunda melhor formiga e assim sucessivamente
        Atualiza valor da célula da matriz pheromone
        '''
        for i in range(len(ant.path) - 1):
            next_node = ant.path[i + 1]
            self.pheromone[current_node, next_node] += factor*ant.of_value
            

    def go_through_path_sum(self, ant, factor):
        '''
        Function to calculate the amount of pheromone to be deposited in trail piece 
        '''
        current_node = ant.path[0]

        '''
        Executa loop por n_ants vezes (0 até n_ants-1)
        Nó inicial é o da melhor formiga
        Nó seguinte é o da segunda melhor formiga e assim sucessivamente
        Atualiza valor da célula da matriz pheromone
        '''
        for i in range(len(ant.path) - 1):
            next_node = ant.path[i + 1]
            self.pheromone[current_node, next_node] += factor*(self.phi * ant.of_value) + ((1-self.phi)*(self.n_ants-len(self.distances))/self.n_ants) 

        
    def update_pheromone(self, best_ant, n_bests_ants):
        '''
        Function to update the pheromone matrix
        '''
        '''
        Multiplica valores da matriz pheronome por rho 0.99)
        Inicialmente a matriz pheromone contém 1 que corresponde ao valor inicial de tau
        '''
        self.pheromone *= (1-self.rho)
        
        for i in range(self.n_best):    # Executa loop por n_best vezes. O valor padrão de n_best é 30% do número de variaveis
            '''
            Executa método go_through_path (calculate the amount of pheromone to be deposited in trail piece) da classe AntColony
            Executa loop por n_best vezes atualizando o valor do feromonio de acordo com o ranking (n_best-i) 
            de cada formiga i
            A atualização é feita na matriz de feromonios
            '''
            self.go_through_path_sum(n_bests_ants[i], (self.n_best - i))
                

        '''
        Executa método go_through_path (calculate the amount of pheromone to be deposited in trail piece) da classe AntColony
        Adiciona quantidade extra de feromonio no caminho percorrido pela melhor formiga de todas
        O valor adicional de feromonio é calculado tomando como fator n_best+1 
        '''
        self.go_through_path(best_ant, self.n_best)

        # Normalizando tabela de feromonios entre [0,1]
        self.pheromone = self.pheromone / self.pheromone.max()
