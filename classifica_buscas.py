import pandas as pd # pd = python data or 'pandas'

# df = data frame
df = pd.read_csv('dist/busca.csv')
X_df = df[['home', 'busca', 'logado']]  # sendo mais de uma, necessario estar num array
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)  # separa as strings transformando em colunas
Ydummies_df = pd.get_dummies(Y_df)['sim']  # separa as strings transformando em colunas

X = Xdummies_df.values  # transforma de dataframe (df) em array
Y = Ydummies_df.values

# Chutar a base entre elementos com 1 ou 0
acerto_um = sum(Y)  # soma os elementos com 1
acerto_zero = len(Y) - acerto_um  # retira os elementos 1 do total de elementos

taxa_acerto_base = max(acerto_um, acerto_zero) / len(Y) * 100.0

print("Taxa de acerto base: %f" % taxa_acerto_base)

porcentagem_treino = 0.9  # definicao da porcentagem que sera para treino

tamanho_treino = int(porcentagem_treino * len(X))  # define os que serao treinados
tamanho_teste = len(Y) - tamanho_treino  # define os que serao testados

treino_dados = X[:tamanho_treino]  # pega os dados para treino
treino_marcacoes = Y[:tamanho_treino]  # pega as marcacoes para treino

teste_dados = X[-tamanho_teste:]  # pega os dados para teste
teste_marcacoes = Y[-tamanho_teste:]  # pega as marcacoes para teste


from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d==0]

total_acertos = len(acertos)
total_elementos = len(teste_dados)
taxa_acerto = 100.0 * total_acertos / total_elementos

print("Taxa de acerto do algoritmo: %f" % taxa_acerto)
print(total_elementos)
