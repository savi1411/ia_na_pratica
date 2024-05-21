# Fundamentos da Inteligência Artificial - Laboratório

## Parte 1: Enunciados

# Exercício 1:
# Implementar uma função de decisão baseada em regras que decide a ação a ser tomada com base no nível de risco.
# - Se o risco for 'alto', a ação deve ser 'evitar'.
# - Se o risco for 'médio', a ação deve ser 'monitorar'.
# - Se o risco for 'baixo', a ação deve ser 'prosseguir'.

# Exercício 2:
# Implementar um Perceptron para a função lógica OR.
# - Treine o Perceptron com dados de treinamento para a função OR.
# - Teste o Perceptron com todos os possíveis valores de entrada (0, 0), (0, 1), (1, 0), (1, 1).

# Estudo de Caso:
# Criar um modelo de classificação para prever se um paciente está 'saudável' ou 'doente' com base em características como idade, pressão arterial e colesterol.
# - Usar um classificador de Árvore de Decisão.
# - Treinar o modelo com dados fictícios fornecidos.
# - Avaliar o modelo com novos dados de pacientes.

## Parte 2: Reservado para as suas soluções

#
##
###
####
#####
######
#######
########
#########
##########
###########
############
#############
##############
###############
################
#################
##################
###################
####################
#####################
######################
#######################
########################
#########################
##########################
###########################
############################
#############################
##############################
###############################
################################
#################################
##################################
###################################
####################################
#####################################
######################################
#######################################
########################################
#########################################
##########################################
###########################################
############################################

## Parte 3: Soluções Sugeridas dos Problemas

# Exercício 1: Função de Decisão Baseada em Regras
# Importando a biblioteca numpy para operações matemáticas
import numpy as np

# Função que decide a ação a ser tomada com base no nível de risco
def decidir_acao(risco):
    if risco == 'alto':
        return 'evitar'
    elif risco == 'médio':
        return 'monitorar'
    elif risco == 'baixo':
        return 'prosseguir'
    else:
        return 'risco desconhecido'

# Testando a função com diferentes níveis de risco
riscos = ['alto', 'médio', 'baixo', 'desconhecido']
for risco in riscos:
    print(f"Risco: {risco} - Ação: {decidir_acao(risco)}")

# Exercício 2: Perceptron para Função Lógica OR
import numpy as np

# Definindo a função de ativação
def step_function(x):
    return 1 if x >= 0 else 0

# Classe que representa um Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)  # Inicializa os pesos
        self.learning_rate = learning_rate

    def predict(self, x):
        z = self.weights.T.dot(np.insert(x, 0, 1))  # Produto escalar
        return step_function(z)  # Aplica a função de ativação

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for i in range(len(y)):
                prediction = self.predict(X[i])
                self.weights += self.learning_rate * (y[i] - prediction) * np.insert(X[i], 0, 1)  # Ajusta os pesos

# Dados de treinamento para a função OR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # Saídas esperadas

# Criando e treinando o Perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

# Testando o Perceptron com os dados de treinamento
print("Testando o Perceptron para a função OR:")
for x in X:
    print(f"{x}: {perceptron.predict(x)}")

# Estudo de Caso: Modelo de Classificação com Árvore de Decisão
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dados fictícios: [idade, pressão arterial, colesterol]
X = np.array([[25, 120, 85], [30, 140, 90], [35, 130, 80], [40, 135, 95], [50, 150, 100], [60, 160, 110]])
y = np.array(['saudável', 'doente', 'saudável', 'doente', 'doente', 'doente'])  # Rótulos

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Criando e treinando o modelo de Árvore de Decisão
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Avaliando o modelo
precisao = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo: {precisao}")

# Testando o modelo com novos dados de pacientes
novos_pacientes = np.array([[45, 145, 92], [55, 155, 105]])
previsoes = modelo.predict(novos_pacientes)
print("Previsões para novos pacientes:")
for i, previsao in enumerate(previsoes):
    print(f"Paciente {i+1}: {previsao}")
