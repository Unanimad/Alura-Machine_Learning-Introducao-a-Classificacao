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
