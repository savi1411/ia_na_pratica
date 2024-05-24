# Fundamentos da Inteligência Artificial - Laboratório

## Parte 1: Enunciados

# Exercício 1:
# Implementar uma função de decisão baseada em regras que decide a ação a ser tomada com base no nível de risco.
# - Se o risco for 'alto', a ação deve ser 'evitar'.
# - Se o risco for 'médio', a ação deve ser 'monitorar'.
# - Se o risco for 'baixo', a ação deve ser 'prosseguir'.

# Exercício 2:
# Reprogramação do exemplo prático de chatbot
# - Utilize como referência o exemplo prático do Teste de Turing do livro.
# - Reprograme o chatbot para que todas as respostas sejam de uma única palavra.
# - Teste o chatbot com diferentes entradas e verifique se as respostas fazem sentido.

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

###############################################

# Exercício 2: Versão do chatbot que responde apenas com uma palavra
def turing_test_one_word():
    print("Chatbot: Olá! Como posso ajudá-lo hoje?")
    while True:
        user_input = input("Você: ")
        if "olá" in user_input.lower() or "oi" in user_input.lower():
            print("Chatbot: Oi.")
        elif "tchau" in user_input.lower() or "adeus" in user_input.lower():
            print("Chatbot: Tchau!")
            break
        elif "nome" in user_input.lower():
            print("Chatbot: Chatbot.")
        elif "tempo" in user_input.lower():
            print("Chatbot: Desconheço.")
        elif "você é um robô" in user_input.lower() or "você é humano" in user_input.lower():
            print("Chatbot: Humano.")
        else:
            print("Chatbot: Repita.")

# Executar o chatbot para teste
turing_test_one_word()

###############################################################

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
