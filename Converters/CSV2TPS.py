''' 
This functión transform the CSV files into TPS files
'''

def csv2tps(csvfile, tpsfile, fname = ''):
    import pandas as pd
    import numpy as np
    archivo = pd.read_csv(f"{csvfile}")
        
    matrix = archivo.to_numpy() 
    nombres = []
    coordenadas = []
    valoresX = matrix[:, 1::2]
    valoresY = matrix[:, 2::2]
 
    for i in matrix:
        nombres.append(i[0])
    for i in range(len(valoresX)):
        
        lista_tuplas = list(zip(valoresX[i],valoresY[i]))
        coordenadas.append(lista_tuplas)
    data = {'id': nombres, 'landmarks': coordenadas}

    with open(tpsfile, 'w') as file:
        contador = 0
        for idx, id in enumerate(data['id']):
            file.write(f"LM={int(len(valoresX[0]))}\n")
            for (x, y) in data['landmarks'][idx]:
                file.write(f"{x} {y}\n")
            file.write(f"IMAGE={id}\n")
            contador += 1
            file.write(f"ID={contador}\n")
    with open(tpsfile, 'w') as file:
        contador = 0
        for idx, id in enumerate(data['id']):
            file.write(f"LM={int(len(valoresX[0]))}\n")
            for (x, y) in data['landmarks'][idx]:
                file.write(f"{x} {y}\n")
            file.write(f"IMAGE={id}\n")
            file.write(f"ID={contador}\n")
            contador += 1