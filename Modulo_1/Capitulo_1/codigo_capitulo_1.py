# Livro: Inteligência Artificial na Prática
# Autor: Carlos Alberto Savi
# Módulo 1: Introdução à Inteligência Artificial
# Capítulo 1: Fundamentos da Inteligência Artificial

# Exemplo básico de uma função que simula uma decisão baseada em regras simples.
# A função decide_action toma uma entrada (input_data) e retorna uma ação baseada na regra definida.

def decide_action(input_data):
    # Se a entrada for 'situação A', retorna 'ação X'
    if input_data == 'situação A':
        return 'ação X'
    # Se a entrada for 'situação B', retorna 'ação Y'
    elif input_data == 'situação B':
        return 'ação Y'
    # Para qualquer outra entrada, retorna 'ação padrão'
    else:
        return 'ação padrão'

# Testando a função com diferentes entradas
print(decide_action('situação A'))  # Saída: ação X
print(decide_action('situação B'))  # Saída: ação Y
print(decide_action('situação C'))  # Saída: ação padrão

#####################################################################################

# Exemplo simples de um chatbot para simular o Teste de Turing
def turing_test():
    print("Chatbot: Olá! Como posso ajudá-lo hoje?")
    while True:
        user_input = input("Você: ")
        if "olá" in user_input.lower() or "oi" in user_input.lower():
            print("Chatbot: Olá! Como você está?")
        elif "tchau" in user_input.lower() or "adeus" in user_input.lower():
            print("Chatbot: Até logo! Tenha um ótimo dia!")
            break
        elif "nome" in user_input.lower():
            print("Chatbot: Eu sou um chatbot. E você, como se chama?")
        elif "tempo" in user_input.lower():
            print("Chatbot: Eu sou apenas um programa, não sei como está o tempo.")
        elif "você é um robô" in user_input.lower() or "você é humano" in user_input.lower():
            print("Chatbot: Eu sou um chatbot, um programa de computador criado para conversar com você.")
        else:
            print("Chatbot: Desculpe, não entendi. Pode repetir?")

# Executar o teste de Turing simples
turing_test()

#####################################################################################

# Exemplo do setor da Saúde - Modelo de Diagnóstico

# Importando a biblioteca scikit-learn, que oferece ferramentas simples e eficientes para análise de dados
from sklearn.tree import DecisionTreeClassifier

# Dados fictícios: [idade, pressão arterial, colesterol]
# Estes dados representam características de pacientes usadas para prever se estão saudáveis ou doentes
X = [[25, 120, 85], [30, 140, 90], [35, 130, 80], [40, 135, 95]]
y = ['saudável', 'doente', 'saudável', 'doente']  # Rótulos correspondentes

# Criando e treinando o modelo de Árvore de Decisão
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Fazendo uma previsão para um novo paciente
novo_paciente = [[32, 128, 88]]
previsão = modelo.predict(novo_paciente)
print(previsão)  # Saída: ['saudável']

#####################################################################################

# Exemplo do setor de Finanças - Previsão do Preço de Ações

# Importando a biblioteca numpy para manipulação de arrays
import numpy as np
# Importando a biblioteca scikit-learn para criar um modelo de regressão linear
from sklearn.linear_model import LinearRegression

# Dados fictícios: [dias, preço]
# Estes dados representam o preço de uma ação ao longo de vários dias
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([100, 101, 102, 103, 104])

# Criando e treinando o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X, y)

# Fazendo uma previsão para o próximo dia
proximo_dia = np.array([6]).reshape(-1, 1)
previsão = modelo.predict(proximo_dia)
print(previsão)  # Saída: [105.]

#####################################################################################

# Exemplo do setor de Tecnologia - Sistema de Recomendação

# Importando a biblioteca scikit-learn para criar um modelo de vizinhos mais próximos
from sklearn.neighbors import NearestNeighbors

# Dados fictícios: [usuário, item]
# Estes dados representam as interações entre usuários e itens (por exemplo, filmes ou produtos)
X = np.array([[1, 101], [1, 102], [1, 103], [2, 101], [2, 102], [3, 103], [3, 104]])

# Criando e treinando o modelo de vizinhos mais próximos
modelo = NearestNeighbors(n_neighbors=2)
modelo.fit(X)

# Recomendando itens para um novo usuário
novo_usuario = np.array([[4, 101]])
distancias, indices = modelo.kneighbors(novo_usuario)
print(indices)  # Saída: [[0 1]]

#####################################################################################

# Exemplo do setor de Educação
# Sistema de tutoria inteligente que sugere recursos de aprendizado com base nas notas dos alunos

def recomendar_recursos(nota):
    # Dependendo da nota do aluno, retorna uma recomendação
    if nota >= 90:
        return "Parabéns! Continue assim. Tente desafios avançados."
    elif nota >= 70:
        return "Bom trabalho! Revise os tópicos principais e pratique mais exercícios."
    else:
        return "Recomendamos revisar os conceitos básicos e fazer exercícios de reforço."

# Testando o sistema com diferentes notas
notas = [95, 85, 60]
for nota in notas:
    print(f"Nota: {nota} - {recomendar_recursos(nota)}")

#####################################################################################

# Exemplo do setor do Direito - Pesquisa Jurídica

# Importando bibliotecas da scikit-learn para análise de texto e classificação
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dados fictícios: [texto jurídico]
# Estes dados representam documentos jurídicos categorizados
documentos = [
    "Contrato de aluguel com prazo de 12 meses.",
    "Processo de divórcio amigável.",
    "Ação de indenização por danos morais.",
    "Recurso de apelação em processo criminal."
]

# Etiquetas correspondentes aos documentos
categorias = ["contrato", "divórcio", "indenização", "criminal"]

# Transformando os dados em vetores de contagem
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documentos)

# Criando e treinando o modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X, categorias)

# Fazendo uma previsão para um novo documento
novo_documento = ["Pedido de indenização por quebra de contrato."]
X_novo = vectorizer.transform(novo_documento)
previsão = modelo.predict(X_novo)
print(previsão)  # Saída: ['contrato']

#####################################################################################

# Exemplo de função que avalia o impacto ético de uma decisão.
# A função avaliar_impacto_etico toma uma decisão e retorna uma avaliação do impacto ético.

def avaliar_impacto_etico(decisao):
    if decisao == 'ação X':
        return 'Alto impacto positivo, baixo risco'
    elif decisao == 'ação Y':
        return 'Impacto positivo moderado, risco moderado'
    else:
        return 'Baixo impacto, baixo risco'

# Avaliando diferentes decisões
print(avaliar_impacto_etico('ação X'))  # Saída: Alto impacto positivo, baixo risco
print(avaliar_impacto_etico('ação Y'))  # Saída: Impacto positivo moderado, risco moderado
print(avaliar_impacto_etico('ação Z'))  # Saída: Baixo impacto, baixo risco
