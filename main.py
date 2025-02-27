import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('saudefinal.csv')

# Função Ver Dataset
def ver_data(df):
    print(df.head())

# Funções Análise Exploratória
def analise_categorica(df, coluna_categoria, categorias, nome_categoria):
    colunas = ['Espontâneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
    resultado = pd.DataFrame(index=categorias.values(), columns=colunas)
    for valor, nome in categorias.items():
        filtro = df[df[coluna_categoria] == valor]
        total = len(filtro)
        if total > 0:
            for coluna in colunas:
                resultado.loc[nome, coluna] = f"{(filtro[coluna].mean() * 100):.1f}%"
        else:
            resultado.loc[nome] = [f"0.0%" for _ in colunas]
    print(f"Análise de {nome_categoria}:\n", resultado, "\n")
# Dicionários de categorias
generos = {0: 'Homens', 1: 'Mulheres', 2: 'Outro'}
estados_civis = {0: 'Sem indicação', 1: 'Solteiro', 2: 'Casado/União Facto', 3: 'Divorciado/Separado', 4: 'Viúvo', 5: 'Outro'}
idades = {1: 'Geração Z', 2: 'Geração Y', 3: 'Geração X', 4: 'Geração BB'}
orientacoes_politicas = {0: 'Sem opção', 1: 'Esquerda', 2: 'Centro', 3: 'Direita'}
situacoes_profissionais = {0: 'Sem indicação', 1: 'Empregado(a)', 2: 'Desempregado(a)', 3: 'Estudante', 4: 'Trabalhador Estudante', 5: 'Reformado(a)', 6: 'Outra'}
graus_escolares = {0: 'Sem indicação', 1: '1º Ciclo', 2: '2º Ciclo', 3: '3º Ciclo', 4: 'Ensino Secundário', 5: 'Bacharelato', 6: 'Licenciatura', 7: 'Mestrado', 8: 'Doutoramento', 9: 'Outro'}
# Chamadas da função para diferentes categorias
def analise_genero(df):
    analise_categorica(df, 'Genero', generos, "Gênero")
def analise_estado_civil(df):
    analise_categorica(df, 'Estado Civil', estados_civis, "Estado Civil")
def analise_idade(df):
    analise_categorica(df, 'Idade', idades, "Idade")
def analise_orientacao_politica(df):
    analise_categorica(df, 'Orient Política', orientacoes_politicas, "Orientação Política")
def analise_situacao_profissional(df):
    analise_categorica(df, 'Situação Profiss', situacoes_profissionais, "Situação Profissional")
def analise_grau_escolar(df):
    analise_categorica(df, 'Grau Escolar', graus_escolares, "Grau Escolar")


# Funções para Visualização de Dados
mapas_valores = {
    'Genero': {0: 'Masculino', 1: 'Feminino', 2: 'Outro'},
    'Idade': {1: 'Geração Z', 2: 'Geração Y', 3: 'Geração X', 4: 'Geração BB'},
    'Estado Civil': {0: 'Sem indicação', 1: 'Solteiro', 2: 'Casado/União Facto', 3: 'Divorciado/Separado', 4: 'Viúvo', 5: 'Outro'},
    'Grau Escolar': {0: 'Sem indicação', 1: '1º Ciclo', 2: '2º Ciclo', 3: '3º Ciclo', 4: 'Ensino Secundário', 5: 'Bacharelato', 6: 'Licenciatura', 7: 'Mestrado', 8: 'Doutoramento', 9: 'Outro'},
    'Situação Profiss': {0: 'Sem indicação', 1: 'Empregado(a)', 2: 'Desempregado(a)', 3: 'Estudante', 4: 'Trabalhador Estudante', 5: 'Reformado(a)', 6: 'Outra'},
    'Orient Política': {0: 'Sem opção', 1: 'Esquerda', 2: 'Centro', 3: 'Direita'}
}
opcoes_colunas = {1: 'Genero', 2: 'Idade', 3: 'Estado Civil', 4: 'Grau Escolar', 5: 'Situação Profiss',
                  6: 'Orient Política', 7: 'Classes'}
def visualizar_pie_chart(df):
    opcoes_colunas_pie = {1: 'Genero', 2: 'Idade', 3: 'Estado Civil', 4: 'Grau Escolar', 5: 'Situação Profiss', 6: 'Orient Política'}
    while True:
        print("\nEscolha a coluna para o Pie Chart:")
        for chave, valor in opcoes_colunas_pie.items():
            print(f"{chave} - {valor}")
        print("0 - Voltar")
        coluna_num = input("Escolha o número da coluna (0 para voltar): ")
        if coluna_num == '0':
            print("Operação cancelada. Voltando ao menu anterior...")
            return
        if not coluna_num.isdigit():
            print("Erro: Por favor, insira um número válido.")
            continue
        coluna_num = int(coluna_num)
        if coluna_num not in opcoes_colunas_pie:
            print("Erro: Opção inválida. Escolha um número da lista.")
            continue
        coluna = opcoes_colunas_pie[coluna_num]
        if coluna in mapas_valores:
            valores_mapeados = df[coluna].map(mapas_valores[coluna])
        dados = valores_mapeados.value_counts()
        plt.figure(figsize=(8, 8))
        dados.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='tab10')
        plt.title(f"Pie Chart: {coluna}")
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
        break

def visualizar_barras_empilhadas(df):
    while True:
        print("\nEscolha as colunas para o gráfico de barras empilhadas:")
        for chave, valor in opcoes_colunas.items():
            print(f"{chave} - {valor}")
        print("0 - Voltar")
        coluna_x_num = input("Escolha o número da coluna para o eixo X: ")
        if coluna_x_num == '0':
            print("Operação cancelada. Voltando ao menu anterior...")
            return
        coluna_y_num = input("Escolha o número da coluna para o empilhamento (diferenciação): ")
        if coluna_y_num == '0':
            print("Operação cancelada. Voltando ao menu anterior...")
            return
        if not coluna_x_num.isdigit() or not coluna_y_num.isdigit():
            print("Erro: Por favor, insira números válidos.")
            continue
        coluna_x_num = int(coluna_x_num)
        coluna_y_num = int(coluna_y_num)
        if coluna_x_num not in opcoes_colunas or coluna_y_num not in opcoes_colunas:
            print("Erro: Opções escolhidas são inválidas. Por favor, escolha números válidos.")
            continue
        coluna_x = opcoes_colunas[coluna_x_num]
        coluna_y = opcoes_colunas[coluna_y_num]
        if coluna_x_num == 7:
            print("Transformação das classes para uso como eixo X...")
            colunas_classes = ['Espontâneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
            df_melted = df.melt(
                id_vars=[coluna_y],
                value_vars=colunas_classes,
                var_name='Classe',
                value_name='Valor'
            )
            df_melted = df_melted[df_melted['Valor'] == 1]
            coluna_x = 'Classe'
            data = df_melted.groupby([coluna_x, coluna_y]).size().unstack(fill_value=0)
        else:
            if coluna_y_num == 7:
                colunas_classes = ['Espontâneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
                print(f"Você escolheu empilhar pelas classes: {', '.join(colunas_classes)}")
                data = df.groupby(coluna_x)[colunas_classes].sum()
            else:
                print(f"Você escolheu: {coluna_x} x {coluna_y}")
                data = df.groupby([coluna_x, coluna_y]).size().unstack(fill_value=0)
        if coluna_x in mapas_valores:
            data.index = data.index.map(mapas_valores[coluna_x])
        if coluna_y in mapas_valores:
            data.columns = data.columns.map(mapas_valores[coluna_y])
        data_percentual = data.div(data.sum(axis=1), axis=0) * 100
        data_percentual.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
        plt.title(f"Barras Empilhadas: {coluna_x} x {coluna_y if coluna_y_num != 7 else 'Classes'}")
        plt.xlabel(coluna_x)
        plt.ylabel("Percentual (%)")
        plt.legend(title=coluna_y if coluna_y_num != 7 else 'Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        break


def visualizar_violin_plot(df):
    df_temp=df.copy()
    while True:
        print("\nEscolha as colunas para o Violin Plot:")
        for chave, valor in opcoes_colunas.items():
            print(f"{chave} - {valor}")
        print("0 - Voltar")
        coluna_x_num = input("Escolha o número da coluna para o eixo X: ")
        if coluna_x_num == '0':
            print("Operação cancelada. Voltando ao menu anterior...")
            return
        coluna_y_num = input("Escolha o número da coluna para o eixo Y ('7 - Classes' não pode ser usada): ")
        if coluna_y_num == '0':
            print("Operação cancelada. Voltando ao menu anterior...")
            return
        if not coluna_x_num.isdigit() or not coluna_y_num.isdigit():
            print("Erro: Por favor, insira números válidos.")
            continue
        coluna_x_num = int(coluna_x_num)
        coluna_y_num = int(coluna_y_num)
        if coluna_x_num not in opcoes_colunas or coluna_y_num not in opcoes_colunas:
            print("Erro: Opções escolhidas são inválidas. Por favor, escolha números válidos.")
            continue
        coluna_x = opcoes_colunas[coluna_x_num]
        coluna_y = opcoes_colunas[coluna_y_num]
        if coluna_y_num == 7:
            print("Erro: 'Classes' não pode ser usada como eixo Y.")
            continue
        if coluna_x == 'Classes':
            print("Transformação das classes para uso como eixo X...")
            df_melted = df_temp.melt(
                id_vars=[coluna_y],
                value_vars=['Espontâneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional'],
                var_name='Classes',
                value_name='Valor'
            )
            coluna_x = 'Classes'
            df_temp = df_melted
        if coluna_x in mapas_valores:
            df_temp[coluna_x] = df_temp[coluna_x].map(mapas_valores[coluna_x])
        if coluna_y in mapas_valores:
            df_temp[coluna_y] = df_temp[coluna_y].map(mapas_valores[coluna_y])
        sns.violinplot(x=coluna_x, y=coluna_y, data=df_temp, hue=coluna_x, palette='tab10', legend=False)
        plt.title(f"Violin Plot: {coluna_x} vs {coluna_y}")
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        break

def visualizar_heatmap(df):
    df_numeric = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmap de Correlação do Dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

# Funções para Classificação
def classificar_racional(df):
    X = df.drop(columns=['Racional'])
    y = df['Racional']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {clf.score(X_test, Y_test):.4f}\n")
    print("Relatório de Classificação:\n", classification_report(Y_test, y_pred))
    cm = confusion_matrix(Y_test, y_pred)
    class_names = ["Não Racional", "Racional"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
    novo_registro = {}
    print("\nDigite os valores para fazer a previsão:")
    for coluna in X.columns:
        valor = input(f"{coluna}: ")
        novo_registro[coluna] = float(valor)
    novo_df = pd.DataFrame([novo_registro])
    previsao = clf.predict(novo_df)[0]
    if previsao == 1:
        print("\nÉ classificado como Racional.")
    else:
        print("\nNÃO é classificado como Racional.")

def classificar_multiplos(df):
    target_columns = ['Espontâneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
    X = df.drop(columns=target_columns)
    Y = df[target_columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    modelo = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    print("\nRelatório de Classificação:\n")
    print(classification_report(Y_test, Y_pred, target_names=target_columns, zero_division=0))
    print("\nDigite os valores para fazer a previsão:")
    novo_registro = {}
    for coluna in X.columns:
        valor = input(f"{coluna}: ")
        novo_registro[coluna] = float(valor)
    novo_df = pd.DataFrame([novo_registro])
    previsao = modelo.predict(novo_df)[0]
    print("\nResultados das previsões:")
    for i, classe in enumerate(target_columns):
        resultado = "Sim" if previsao[i] == 1 else "Não"
        print(f"- {classe}: {resultado}")

# Função Menu Principal
def menu():
    while True:
        print("\nMenu Principal:")
        print("1 - Ver Dataset")
        print("2 - Análise Exploratória")
        print("3 - Visualização de Dados")
        print("4 - Classificação 'Racional'")
        print("5 - Classificação Multi-Classe")
        print("0 - Sair")
        escolha = input("Escolha uma opção: ")
        if escolha == '1':
            ver_data(data)
        elif escolha == '2':
            analise_exploratoria()
        elif escolha == '3':
            visualizacao_dados()
        elif escolha == '4':
            classificar_racional(data)
        elif escolha == '5':
            classificar_multiplos(data)
        elif escolha == '0':
            print("Saindo do programa.")
            break
        else:
            print("Opção inválida, tente novamente.")

# Função Menu Análise Exploratória
def analise_exploratoria():
    while True:
        print("\n1 - Análise Exploratória")
        print("   1 - Género")
        print("   2 - Estado Civil")
        print("   3 - Idades")
        print("   4 - Orientação Política")
        print("   5 - Situação Profissional")
        print("   6 - Grau Escolar")
        print("0 - Voltar")
        escolha = input("Escolha uma opção: ")
        if escolha == '1':
            analise_genero(data)
        elif escolha == '2':
            analise_estado_civil(data)
        elif escolha == '3':
            analise_idade(data)
        elif escolha == '4':
            analise_orientacao_politica(data)
        elif escolha == '5':
            analise_situacao_profissional(data)
        elif escolha == '6':
            analise_grau_escolar(data)
        elif escolha == '0':
            break
        else:
            print("Opção inválida, tente novamente.")

# Função Menu Visualização de Dados
def visualizacao_dados():
    while True:
        print("\n2 - Visualização de Dados")
        print("   1 - Pie Charts")
        print("   2 - Barras Empilhadas")
        print("   3 - Violin Plot")
        print("   4 - Heatmap")
        print("0 - Voltar")
        escolha = input("Escolha uma opção: ")
        if escolha == '1':
            visualizar_pie_chart(data)
        elif escolha == '2':
            visualizar_barras_empilhadas(data)
        elif escolha == '3':
            visualizar_violin_plot(data)
        elif escolha == '4':
            visualizar_heatmap(data)
        elif escolha == '0':
            break
        else:
            print("Opção inválida, tente novamente.")

# Chamar o menu
menu()
