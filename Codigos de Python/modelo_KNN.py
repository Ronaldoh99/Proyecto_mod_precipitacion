#prueba que funciona python desde VScod
print("HOLA MUNDO")
print("aqui va mi codigo por linea")
#para ejecutar linea por linea shift + enter 
!pip install pandasgui

#lectura de librerias o paquetes 
import pandas as pd
from pandas import DataFrame
from IPython.display import display


import pandas as pd
from pandasgui import show

import numpy as np
import os 
import matplotlib.pyplot as plt

# automatizacion de lectura de paquetes 
import subprocess
import sys
import importlib

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

# Lista de paquetes a instalar e importar
#packages = ['pandas', 'IPython', 'pandasgui', 'numpy', 'matplotlib','joblib','cudf','cupy']
packages = ['pandas', 'IPython', 'pandasgui', 'numpy', 'matplotlib','joblib']

for package in packages:
    install_and_import(package)

from pandas import DataFrame
from IPython.display import display
from pandasgui import show

import os
import matplotlib.pyplot as plt


#

#observar donde estoy en ruta 
os.getcwd()

#lectura de datos 
#datos = pd.read_csv("Datos/datos_combinados_para_modelo.txt",sep=";")
datos = pd.read_csv("Datos/datos_seleccionados_para_modelo_coordenadas.txt",sep=";")
#str en R o tipo de datos 
datos.dtypes
print("los tipos de datos son: ",datos.dtypes)
datos.info()

#ver datos en consola
display(datos)
#ver datos de forma iteractiva en una pantalla extra de la extension de pandas
show(datos)

#pruebas del modelo KNN 
#scikit-learn es una biblioteca de aprendizaje automático de código 
# abierto que ofrece implementaciones eficientes de una amplia variedad
#  de algoritmos de aprendizaje, incluyendo KNN. Además, proporciona funciones
#  para la imputación de datos faltantes utilizando KNN y otras técnicas.


#PRUEBAS CON POCOS DATOS SIMULANDO ############NO EJECUTAR###################
datos.columns

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['precipitation']])
    station_data['precipitation_imputed'] = station_data_imputed
    return station_data

# Crear datos de ejemplo
data = pd.DataFrame({
    'station_id': [1, 1, 2, 2, 3, 3],
    'precipitation': [1, np.nan, 3, 4, np.nan, 6]
})

stations = data['station_id'].unique()
imputed_data = pd.DataFrame()

for station in stations:
    station_data = data[data['station_id'] == station]
    station_data_imputed = impute_precipitation_for_station(station_data)
    imputed_data = imputed_data.append(station_data_imputed)

print(imputed_data)

###################   Fin de prueba con pocos datos ################################

# Crear un nuevo DataFrame con nombres de columnas cambiados
data = datos.rename(columns={'Date': 'date', 'prec': 'precipitation','ID':'station_id'})
data.info()
#
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed


def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['precipitation']])
    station_data['precipitation_imputed'] = station_data_imputed
    return station_data

#El parámetro n_jobs controla el número de procesos en paralelo. Si n_jobs=-1, 
#se utilizarán todos los núcleos disponibles en tu computadora.
def impute_precipitation_parallel(data, n_jobs=4):
    stations = data['station_id'].unique()
    imputed_data = pd.concat(
        Parallel(n_jobs=n_jobs)(
            delayed(impute_precipitation_for_station)(data[data['station_id'] == station])
            for station in stations
        ),
        ignore_index=True,
    )
    return imputed_data


imputed_data = impute_precipitation_parallel(data)

#La función impute_precipitation_parallel ejecutará la imputación de datos de precipitación 
# utilizando KNN Imputer en paralelo para todas las estaciones en tus datos. 
# El parámetro n_jobs controla el número de procesos en paralelo. 
# Si n_jobs=-1, se utilizarán todos los núcleos disponibles en tu computadora.

import multiprocessing as mp

def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['precipitation']])
    station_data['precipitation_imputed'] = station_data_imputed
    return station_data

def impute_precipitation_parallel(data, n_jobs=4):
    stations = data['station_id'].unique()
    pool = mp.Pool(processes=n_jobs)
    results = [
        pool.apply_async(impute_precipitation_for_station, args=(data[data['station_id'] == station],))
        for station in stations
    ]
    imputed_data = pd.concat([result.get() for result in results], ignore_index=True)
    pool.close()
    pool.join()
    return imputed_data

imputed_data = impute_precipitation_parallel(data)
print(imputed_data)




#prueba a ver si funciona: fUNCIONA 

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed

def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['prec']])
    station_data['prec_imputed'] = station_data_imputed
    return station_data

def impute_precipitation_parallel(data, n_jobs=-1):
    stations = data['ID'].unique()
    results = Parallel(n_jobs=n_jobs)(
        delayed(impute_precipitation_for_station)(data[data['ID'] == station]) for station in stations
    )
    imputed_data = pd.concat(results, ignore_index=True)
    return imputed_data


data = pd.read_csv("Datos/datos_seleccionados_para_modelo_coordenadas.txt",sep=";")
data.info()

#datos de prueba
data = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=5, freq='D').tolist() * 3,
    'prec': [1, np.nan, 3, 4, np.nan] * 3,
    'ID': [1] * 5 + [2] * 5 + [3] * 5,
    'LAT': [12.34] * 5 + [56.78] * 5 + [90.12] * 5,
    'LON': [-76.54] * 5 + [123.45] * 5 + [167.89] * 5
})

data.dtypes
data['Date'] = pd.to_datetime(data['Date']) 
data['prec '] = pd.to_numeric(data['prec']) 
data['LAT '] = float(data['LAT']) 
data['LON '] = float(data['LON']) 
data.dtypes

imputed_data = impute_precipitation_parallel(data)
print(imputed_data)
imputed_data.info()


#####prueba 2.0  funciona con todos los datos 

#instalar a la fuerza 
####prueba en paralelo funciona para datos de prueba
#toca instalar forzadamente en la terminal pip install dask
#la cual es una biblioteca de python que permite de manera flexible
#realizar computacion paralela o procesamientos de informacion en paralelos
#compuesta de 2 partes :
#por una parte "Dynamic task scheduling" o programacion dinamica de tareas
#que consta de una orquestacion de los procesos que se estan realizando 
#por parte la libreria para optimizar el calculo reduciendo el uso de memoria
#y acortando los tiempos 
#
#Aquí hay una comparación detallada de los dos enfoques de imputación en paralelo:
#
#Enfoque 1: joblib
#
#Librerías utilizadas:
#
#joblib: Esta biblioteca se utiliza para realizar cálculos en paralelo y optimizar la ejecución de bucles en Python. Ofrece una API simple y consistente para la ejecución en paralelo y la gestión de la memoria caché.
#Características clave:
#
#Se usa Parallel y delayed de joblib para ejecutar la función impute_precipitation_for_station en paralelo para cada estación.
#El parámetro n_jobs controla el número de núcleos de CPU utilizados en la ejecución en paralelo. Si n_jobs = -1, se utilizarán todos los núcleos disponibles.
#Enfoque 2: dask
#
#Librerías utilizadas:
#
#dask: Es una biblioteca de Python flexible y optimizada para el procesamiento en paralelo y la computación fuera del núcleo. Proporciona estructuras de datos paralelas y una API familiar inspirada en Pandas.
#Características clave:
#
#Se utiliza dask.dataframe para crear un DataFrame de Dask, que es una estructura de datos paralela similar a un DataFrame de Pandas.
#El parámetro npartitions controla cuántas particiones se deben crear al dividir el DataFrame de Pandas en partes más pequeñas.
#En lugar de usar joblib, el enfoque en sí mismo divide el DataFrame en estaciones y ejecuta la imputación para cada estación en las particiones de Dask.
#Se utiliza el método compute() para convertir los resultados del DataFrame de Dask de vuelta a un DataFrame de Pandas después de la imputación.
#Comparación:
#
#Ambos enfoques se centran en mejorar la eficiencia al imputar datos en paralelo, pero utilizan diferentes bibliotecas y métodos para lograrlo.
#
#El enfoque 1 (joblib) es más simple y fácil de entender, pero podría ser menos eficiente en el manejo de la memoria y la paralelización, especialmente si se trabaja con grandes conjuntos de datos.
#
#El enfoque 2 (dask) es más avanzado y ofrece una mayor flexibilidad en el control de la paralelización y la optimización de la memoria. Esto puede resultar en un mejor rendimiento en el caso de conjuntos de datos más grandes o computadoras con recursos limitados. Sin embargo, puede ser un poco más difícil de entender y configurar correctamente.
#
#En resumen, la elección del enfoque depende de tus requisitos de rendimiento y del tamaño de los datos que estés procesando. Si estás trabajando con conjuntos de datos muy grandes, el enfoque de Dask podría ser más adecuado. Si la simplicidad es tu principal preocupación y los conjuntos de datos no son excesivamente grandes, entonces el enfoque de joblib podría ser suficiente.



import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed
import dask.dataframe as dd

def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['prec']])
    
    # Crear un nuevo DataFrame en lugar de modificar el existente
    imputed_df = station_data.copy()
    imputed_df['prec_imputed'] = station_data_imputed
    
    return imputed_df



def impute_precipitation_parallel(data, npartitions=10):
    ddata = dd.from_pandas(data, npartitions=npartitions)
    stations = data['ID'].unique()
    
    results = []
    for station in stations:
        station_data = ddata[ddata['ID'] == station].compute()
        
        # Check if there are enough valid values to perform imputation
        if station_data['prec'].notna().sum() >= 1:
            station_data_imputed = impute_precipitation_for_station(station_data)
            results.append(station_data_imputed)
        else:
            results.append(station_data)
    
    imputed_data = pd.concat(results, ignore_index=True)
    return imputed_data



data = pd.read_csv("Datos/datos_combinados_para_modelo_con_coordenadas.txt",sep=";")
data.info()


imputed_data = impute_precipitation_parallel(data)
print(imputed_data)

imputed_data.info()

soloNAn=na_rows = imputed_data[imputed_data['prec'].isna()]
soloNAn
#  64312  NUMERO DE IMPUTACIONES QUE REALIZO

data.describe['prec']
summary = data['prec'].describe()
print(summary)

#como es la informacion antes de la imputacion por estaciones
print(data.info())
#agrupada por id que sacar el resumen por estaciones 
print(data.groupby('ID').describe())
#Después de la imputación, analizaré el DataFrame imputed_data para evaluar
#el rendimiento del modelo de imputación:
print(imputed_data.info())
print(imputed_data.groupby('ID').describe())


############################## Evaluacion del los modelos KNN por medio de MAE,RMSE,R^2#######################



import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Dividir los datos en conjuntos de entrenamiento y prueba usando solo la columna 'prec'
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['ID'])

# Realizar la imputación en todo el conjunto de datos de entrenamiento
imputed_train_data = impute_precipitation_parallel(train_data)

# Realizar la imputación en todo el conjunto de datos de prueba
imputed_test_data = impute_precipitation_parallel(test_data)

# Calcular el rendimiento de la imputación para cada estación en el conjunto de prueba
# Calcular el rendimiento de la imputación para cada estación en el conjunto de prueba
evaluation_results = {}
for station_id in test_data['ID'].unique():
    station_test_data = imputed_test_data[imputed_test_data['ID'] == station_id]
    
    # Filtrar solo las filas que contienen valores imputados en la columna 'prec'
    imputed_rows = station_test_data[station_test_data['prec'].isna()]
    
    if not imputed_rows.empty:
        # Combinar el conjunto de datos imputado con el conjunto de datos original de prueba para recuperar los valores reales
        real_and_imputed = imputed_rows[['ID', 'Date', 'prec_imputed']].merge(test_data[['ID', 'Date', 'prec']], on=['ID', 'Date'], how='left')

        # Eliminar filas con valores NaN en 'prec_imputed' o 'prec'
        real_and_imputed = real_and_imputed.dropna(subset=['prec_imputed', 'prec'])
        
        if not real_and_imputed.empty:
            # Comparar los valores imputados con los valores originales en el conjunto de prueba
            imputed_values = real_and_imputed['prec_imputed']
            real_values = real_and_imputed['prec']
            
            # Evaluar el rendimiento de la imputación
            rmse, r2, mae = evaluate_imputation_performance(real_values, imputed_values)
            
            evaluation_results[station_id] = {
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae
            }
        else:
            evaluation_results[station_id] = {
                'RMSE': None,
                'R2': None,
                'MAE': None
            }
    else:
        evaluation_results[station_id] = {
            'RMSE': None,
            'R2': None,
            'MAE': None
        }

# Imprimir los resultados de la evaluación
for station_id, performance in evaluation_results.items():
    print(f"Estación {station_id}:")
    if performance['RMSE'] is not None:
        print(f"  RMSE: {performance['RMSE']:.3f}")
        print(f"  R2: {performance['R2']:.3f}")
        print(f"  MAE: {performance['MAE']:.3f}")
    else:
        print("  No hay valores faltantes en los datos de prueba para esta estación.")
    print()


data.info()


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Aplicar KNNImputer en todo el conjunto de datos de entrenamiento
imputed_train_data = impute_precipitation_parallel(train_data)

# Aplicar KNNImputer en todo el conjunto de datos de prueba
imputed_test_data = impute_precipitation_parallel(test_data)

# Filtrar las filas del conjunto de prueba que originalmente tenían valores faltantes en 'prec'
test_data_with_missing = test_data[test_data['prec'].isna()]

# Obtener solo las filas correspondientes en el conjunto de datos imputados
imputed_test_data_with_missing_values = imputed_test_data[imputed_test_data.index.isin(test_data_with_missing.index)]

# Imprimir el conjunto de datos con valores imputados
print(imputed_test_data_with_missing_values)

# Comparar los valores imputados con los valores reales en 'prec' para calcular RMSE y MAE
y_true = test_data_with_missing['prec']
y_pred = imputed_test_data_with_missing_values.loc[y_true.index, 'prec_imputed']

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)







###################PRUEBA PARA PARALELIZAR POR MEDIO DE GPU NVIDIA #############
########   SE DEBE CONTAR UNA RTX DE LA SERIE 3000                 ##############
#parelizacion con gpu no funciona 
import cudf
import cupy as cp

from cuml.experimental.preprocessing import KNNImputer

def impute_precipitation_for_station(station_data, n_neighbors=5):
    station_data_gpu = cudf.DataFrame.from_pandas(station_data)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data_gpu[['precipitation']])
    station_data['precipitation_imputed'] = cp.asnumpy(station_data_imputed.values).flatten()
    return station_data

def impute_precipitation_parallel(data):
    stations = data['station_id'].unique()
    imputed_data = pd.concat(
        [impute_precipitation_for_station(data[data['station_id'] == station]) for station in stations],
        ignore_index=True,
    )
    return imputed_data

# Crear datos de ejemplo
data = pd.DataFrame({
    'station_id': [1, 1, 2, 2, 3, 3],
    'precipitation': [1, np.nan, 3, 4, np.nan, 6]
})

imputed_data = impute_precipitation_parallel(data)
print(imputed_data)