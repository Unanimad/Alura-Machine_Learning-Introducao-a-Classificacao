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

    print('Taxa de acerto do {0}: {1}'.format(nome, taxa_acerto))
    return taxa_acerto

# df = data frame
df = pd.read_csv('dist/busca2.csv')
X_df = df[['home', 'busca', 'logado']]  # sendo mais de uma, necessario estar num array
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)  # separa as strings transformando em colunas
Ydummies_df = Y_df

X = Xdummies_df.values  # transforma de dataframe (df) em array
Y = Ydummies_df.values

porcentagem_treino = 0.8  # definicao da porcentagem de treino
porcentagem_teste = 0.1  # definicao da porcentagem de teste

tamanho_treino = int(porcentagem_treino * len(X))  # define os que serao treinados
tamanho_teste = int(porcentagem_teste * len(Y))  # define os que serao testados
tamanho_validacao = len(Y) - tamanho_treino - tamanho_teste  # define os que serao para validacao

fim_treino = tamanho_treino + tamanho_teste
treino_dados = X[0:tamanho_treino]  # pega os dados para treino
treino_marcacoes = Y[0:tamanho_treino]  # pega as marcacoes para treino

fim_teste = tamanho_treino + tamanho_teste
teste_dados = X[tamanho_treino:fim_teste]  # pega os dados para teste
teste_marcacoes = Y[tamanho_treino:fim_teste]  # pega as marcacoes para teste

validacao_dados = X[fim_teste:]
validacao_marcacoes = Y[fim_teste:]


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict('MultinomialNB', modeloMultinomial,
    treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict('AdaBoostClassifier', modeloAdaBoost,
    treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

# define qual o melhor algoritmo
if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)  # preve

acertos = (resultado == validacao_marcacoes)

total_acertos = sum(acertos)  # soma todos os elementos
total_elementos = len(validacao_marcacoes)
taxa_acerto = 100.0 * total_acertos / total_elementos

print('Taxa de acerto entre os dois algoritmos no mundo real: {0}'"'.format(taxa_acerto))

# Chutar com o elemento mais recorrente
acerto_base = max(Counter(validacao_marcacoes).values())  # retorna o elemento com mais recorrente
taxa_acerto_base = acerto_base / len(validacao_marcacoes) * 100.0

print('Taxa de acerto base: %f' % taxa_acerto_base)
print('Total de testes: %d' % len(validacao_dados))
