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
datos = pd.read_csv("Datos/datos_combinados_para_modelo.txt",sep=";")
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
datos.columns

from sklearn.impute import KNNImputer

def impute_precipitation_for_station(station_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['precipitation']])
    station_data['precipitation_imputed'] = station_data_imputed
    return station_data


stations = datos['ID'].unique()
imputed_data = pd.DataFrame()

for station in stations:
    station_data = datos[datos['ID'] == station]
    station_data_imputed = impute_precipitation_for_station(station_data)
    imputed_data = imputed_data.append(station_data_imputed)

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
