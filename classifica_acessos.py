from acesso import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

# Separa dados para que possam ser feito testes e treinos
treino_dados = X[:90]  # 90 elementos
teste_dados = X[-9:]  # 9 ultimos elementos

treino_marcacoes = Y[:90]
teste_marcacoes = Y[-9:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)  # utiliza os arrays de treino

resultado = modelo.predict(teste_dados)  # preve utilizando os dados de teste
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d==0]
total_acertos = len(acertos)
total_elementos = len(teste_dados)
taxa_acerto = 100.0 * (total_acertos/total_elementos)

print(taxa_acerto)
print(total_elementos)
