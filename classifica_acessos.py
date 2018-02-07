from acesso import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

modelo = MultinomialNB()
modelo.fit(X, Y)

resultado = modelo.predict(X)
diferencas = resultado - Y

acertos = [d for d in diferencas if d==0]
total_acertos = len(acertos)
total_elementos = len(X)
taxa_acerto = 100.0 * (total_acertos/total_elementos)

print(taxa_acerto)
print(total_elementos)
