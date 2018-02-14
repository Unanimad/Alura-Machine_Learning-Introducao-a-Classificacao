from collections import Counter
import pandas as pd # pd = python data or 'pandas'

# Treina e preve
def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = (resultado == teste_marcacoes)

    total_acertos = sum(acertos)  # soma todos os elementos
    total_elementos = len(teste_dados)
    taxa_acerto = 100.0 * total_acertos / total_elementos

    print("Taxa de acerto do {0}: {1}".format(nome, taxa_acerto))

# df = data frame
df = pd.read_csv('dist/busca2.csv')
X_df = df[['home', 'busca', 'logado']]  # sendo mais de uma, necessario estar num array
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)  # separa as strings transformando em colunas
Ydummies_df = Y_df

X = Xdummies_df.values  # transforma de dataframe (df) em array
Y = Ydummies_df.values

porcentagem_treino = 0.9  # definicao da porcentagem que sera para treino

tamanho_treino = int(porcentagem_treino * len(X))  # define os que serao treinados
tamanho_teste = len(Y) - tamanho_treino  # define os que serao testados

treino_dados = X[:tamanho_treino]  # pega os dados para treino
treino_marcacoes = Y[:tamanho_treino]  # pega as marcacoes para treino

teste_dados = X[-tamanho_teste:]  # pega os dados para teste
teste_marcacoes = Y[-tamanho_teste:]  # pega as marcacoes para teste


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

modelo = MultinomialNB()
fit_and_predict('MultinomialNB', modelo, treino_dados, treino_marcacoes,
    teste_dados, teste_marcacoes)

modelo = AdaBoostClassifier()
fit_and_predict('AdaBoostClassifier', modelo, treino_dados, treino_marcacoes,
    teste_dados, teste_marcacoes)

# Chutar com o elemento mais recorrente
acerto_base = max(Counter(teste_marcacoes).values())  # retorna o elemento com mais recorrente
taxa_acerto_base = acerto_base / len(teste_marcacoes) * 100.0

print('Taxa de acerto base: %f' % taxa_acerto_base)
print('Total de testes: %d' % tamanho_teste)
