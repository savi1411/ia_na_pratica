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
