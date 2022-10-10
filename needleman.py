import numpy as np 
from IPython.display import display
import pandas as pd

#Funcao com que retorna o valor do score de match/mismatch da tabela BLOSUM62
def score(pair1, pair2):
    BLOSUM62={'C':{'C':9,'S':-1,'T':-1,'P':-3,'A':0,'G':-3,'N':-3,'D':-3,'E':-4,'Q':-3,'H':-3,'R':-3,'K':-3,'M':-1,'I':-1,'L':-1,'V':-1,'F':-2,'Y':-2,'W':-2},
            'S':{'C':-1,'S':4,'T':1,'P':-1,'A':1,'G':0,'N':1,'D':0,'E':0,'Q':0,'H':-1,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
            'T':{'C':-1,'S':1,'T':4,'P':1,'A':-1,'G':1,'N':0,'D':1,'E':0,'Q':0,'H':0,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
            'P':{'C':-3,'S':-1,'T':1,'P':7,'A':-1,'G':-2,'N':-1,'D':-1,'E':-1,'Q':-1,'H':-2,'R':-2,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-4,'Y':-3,'W':-4},
            'A':{'C':0,'S':1,'T':-1,'P':-1,'A':4,'G':0,'N':-1,'D':-2,'E':-1,'Q':-1,'H':-2,'R':-1,'K':-1,'M':-1,'I':-1,'L':-1,'V':-2,'F':-2,'Y':-2,'W':-3},
            'G':{'C':-3,'S':0,'T':1,'P':-2,'A':0,'G':6,'N':-2,'D':-1,'E':-2,'Q':-2,'H':-2,'R':-2,'K':-2,'M':-3,'I':-4,'L':-4,'V':0,'F':-3,'Y':-3,'W':-2},
            'N':{'C':-3,'S':1,'T':0,'P':-2,'A':-2,'G':0,'N':6,'D':1,'E':0,'Q':0,'H':-1,'R':0,'K':0,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-4},
            'D':{'C':-3,'S':0,'T':1,'P':-1,'A':-2,'G':-1,'N':1,'D':6,'E':2,'Q':0,'H':-1,'R':-2,'K':-1,'M':-3,'I':-3,'L':-4,'V':-3,'F':-3,'Y':-3,'W':-4},
            'E':{'C':-4,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':2,'E':5,'Q':2,'H':0,'R':0,'K':1,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-3},
            'Q':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':0,'E':2,'Q':5,'H':0,'R':1,'K':1,'M':0,'I':-3,'L':-2,'V':-2,'F':-3,'Y':-1,'W':-2},
            'H':{'C':-3,'S':-1,'T':0,'P':-2,'A':-2,'G':-2,'N':1,'D':1,'E':0,'Q':0,'H':8,'R':0,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-1,'Y':2,'W':-2},
            'R':{'C':-3,'S':-1,'T':-1,'P':-2,'A':-1,'G':-2,'N':0,'D':-2,'E':0,'Q':1,'H':0,'R':5,'K':2,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
            'K':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':-1,'E':1,'Q':1,'H':-1,'R':2,'K':5,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
            'M':{'C':-1,'S':-1,'T':-1,'P':-2,'A':-1,'G':-3,'N':-2,'D':-3,'E':-2,'Q':0,'H':-2,'R':-1,'K':-1,'M':5,'I':1,'L':2,'V':-2,'F':0,'Y':-1,'W':-1},
            'I':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-3,'R':-3,'K':-3,'M':1,'I':4,'L':2,'V':1,'F':0,'Y':-1,'W':-3},
            'L':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-4,'E':-3,'Q':-2,'H':-3,'R':-2,'K':-2,'M':2,'I':2,'L':4,'V':3,'F':0,'Y':-1,'W':-2},
            'V':{'C':-1,'S':-2,'T':-2,'P':-2,'A':0,'G':-3,'N':-3,'D':-3,'E':-2,'Q':-2,'H':-3,'R':-3,'K':-2,'M':1,'I':3,'L':1,'V':4,'F':-1,'Y':-1,'W':-3},
            'F':{'C':-2,'S':-2,'T':-2,'P':-4,'A':-2,'G':-3,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-1,'R':-3,'K':-3,'M':0,'I':0,'L':0,'V':-1,'F':6,'Y':3,'W':1},
            'Y':{'C':-2,'S':-2,'T':-2,'P':-3,'A':-2,'G':-3,'N':-2,'D':-3,'E':-2,'Q':-1,'H':2,'R':-2,'K':-2,'M':-1,'I':-1,'L':-1,'V':-1,'F':3,'Y':7,'W':2},
            'W':{'C':-2,'S':-3,'T':-3,'P':-4,'A':-3,'G':-2,'N':-4,'D':-4,'E':-3,'Q':-2,'H':-2,'R':-3,'K':-3,'M':-1,'I':-3,'L':-2,'V':-3,'F':1,'Y':2,'W':11}
            }
    if pair1 and pair2 in BLOSUM62:
        return BLOSUM62[pair1][pair2]
    else:
        return BLOSUM62[tuple(reversed(pair1, pair2))]



#Funcao para montar uma tabela com display das bases   
def table(data_array, row_labels,col_labels):
    df = pd.DataFrame(data_array,index=row_labels,columns=col_labels)
    return df



def main():
    #Entra com as sequencias
    sequence_1 = input("Sequencia 1:")
    sequence_2 = input("Sequencia 2:")

    #Cria matrizes 
    main_matrix = np.zeros((len(sequence_1)+1, len(sequence_2)+1)) #Matrizes de 0s inicial com dimensoes n+1 x m+1
    match_checker_matrix = np.zeros((len(sequence_1), len(sequence_2))) #Matriz nxm que define os matchs e mismatches
    traceback_matrix = np.full([len(sequence_1)+1, len(sequence_2)+1],"-") #Matriz que mostra o caminhamento final

    #Definicoes do Python 3 para gerar as setinhas (apenas visual)
    up_arrow = "\u2191"
    left_arrow = "\u2190"
    up_left_arrow = "\u2196"
    arrow = "-"

    #Scores de match, mismatch (definidos pela matriz Blosum62) e gap
    gap_penalty = int(input("Penalidade de Gap:"))
    for i in range(len(sequence_1)):
        for j in range(len(sequence_2)):
            match_checker_matrix[i][j] = score(sequence_1[i],sequence_2[j])
    
    #Usando Needleman_Wunsch preenche-se a matriz principal
    #Passo 1: Inicializacao
    for i in range(1,len(sequence_1)+1):
        main_matrix[i][0] = i * gap_penalty
        traceback_matrix[i][0] = up_arrow
    for j in range(1,len(sequence_2)+1):
        main_matrix[0][j] = j * gap_penalty
        traceback_matrix[0][j] = left_arrow
    
    #Passo 2: Preencher as matrizes de pontuacao e de caminho
    for i in range(1,len(sequence_1)+1):
        for j in range(1,len(sequence_2)+1):
            main_matrix[i][j] = max(main_matrix[i-1][j-1]+match_checker_matrix[i-1][j-1],
                                    main_matrix[i-1][j]+gap_penalty,
                                    main_matrix[i][j-1]+gap_penalty)
            if i == 0 and j == 0:
                arrow = "-"
            elif i == 0:
                arrow = left_arrow
            elif j == 0:
                arrow = up_arrow
            else: 
                if main_matrix[i][j] == main_matrix[i][j-1]+gap_penalty:
                    arrow = left_arrow
                elif main_matrix[i][j] == main_matrix[i-1][j]+gap_penalty:
                    arrow = up_arrow
                elif main_matrix[i][j] == main_matrix[i-1][j-1]+match_checker_matrix[i-1][j-1]:
                    arrow = up_left_arrow
            traceback_matrix[i,j]= arrow 

    #Passo 3: Traceback
    aligned_1 = ""
    aligned_2 = ""
    ti = len(sequence_1)
    tj = len(sequence_2)
    while(ti >0 or tj > 0):
        if (ti >0 and tj > 0 and main_matrix[ti][tj] == main_matrix[ti-1][tj-1]+ match_checker_matrix[ti-1][tj-1]):
            aligned_1 = sequence_1[ti-1] + aligned_1
            aligned_2 = sequence_2[tj-1] + aligned_2
            ti = ti - 1
            tj = tj - 1
        elif(ti > 0 and main_matrix[ti][tj] == main_matrix[ti-1][tj] + gap_penalty):
            aligned_1 = sequence_1[ti-1] + aligned_1
            aligned_2 = "-" + aligned_2
            ti = ti -1
        else:
            aligned_1 = "-" + aligned_1
            aligned_2 = sequence_2[tj-1] + aligned_2
            tj = tj - 1

    #Impressoes
    #Impressao da tabela de alinhamento
    print("\nTabela de pontuacao final: ")
    row_labels = [label for label in "-"+sequence_1]
    column_labels = [label for label in "-"+sequence_2]
    display(table(main_matrix,row_labels,column_labels))

    #Impressao da tabela de caminhos final:
    print("\nTabela de caminho final: ")
    display(table(traceback_matrix,row_labels,column_labels))

    #Impressao dos alinhamentos feitos com traceback
    print("\nAlinhamento final a partir do traceback da tabela: ")
    print(aligned_1)
    print(aligned_2)


if __name__ == "__main__":
    main()