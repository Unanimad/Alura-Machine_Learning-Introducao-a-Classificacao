caracteristicas = ['gordinho', 'perna_curta', 'auau']

# 1 = sim / 0 = nao
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

cao1 = [1, 1, 1]
cao2 = [0, 1, 1]
cao3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cao1, cao2, cao3]

# Marcação para definir se é porco ou cachorro
# 1 = porco / -1 = cao
marcacoes = [1, 1, 1, -1, -1, -1]

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]

teste = [misterioso1, misterioso2]  # elementos que serao testados

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()  # cria um modelo
modelo.fit(dados, marcacoes)  # adequa o modelo as marcacoes

print(modelo.predict(teste))  # preve qual o elemento
