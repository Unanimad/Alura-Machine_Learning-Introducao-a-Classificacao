import csv

# Abre arquivo CSV e faz a leitura das linhas
def carregar_acessos():
    X = [] # dados
    Y = [] # marcacoes

    arquivo = open('dist/acesso.csv', 'r')  # abre o arquivo CSV
    leitor = csv.reader(arquivo, delimiter=',')  # le o arquivo CSV
    next(leitor)  # pula para proxima linha por causa do nome das colunas

    for home, como_funciona, contato, comprou in leitor:
        dado = [int(home), int(como_funciona), int(contato)]

        # Salva nos arrays
        X.append(dado)
        Y.append(int(comprou))

    return X, Y


def carregar_buscas():
    X = []
    Y = []

    arquivo = open('dist/busca2.csv', 'r')
    leitor = csv.reader(arquivo)
    next(leitor)

    for home, busca, logado, comprou in leitor:
        dado = [int(home), busca, int(logado)]

        X.append(dado)
        Y.append(int(comprou))

    return X, Y
