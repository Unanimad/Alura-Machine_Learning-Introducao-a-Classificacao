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

# animais que nao conhecemos, mas sabemos as caracteristicas
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

testes = [misterioso1, misterioso2, misterioso3]  # elementos que serao testados
marcacoes_teste = [-1, 1, -1]  # garantia de teste, resultado esperado

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()  # cria um modelo
modelo.fit(dados, marcacoes)  # adequa o modelo as marcacoes

resultado = modelo.predict(testes)  # preve qual o elemento
print(resultado)

# verifica as diferencas
diferencas = resultado - marcacoes_teste  # se for diferente de 0 entao errou
print(diferencas)

# verifica os acertos
acertos = [d for d in diferencas if d==0]  # retorna um array

total_elementos = len(testes)  # total de elementos
total_acertos = len(acertos)  # total de acertos

print(100.0*(total_acertos/total_elementos))
