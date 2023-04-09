import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import multiprocessing as mp

def impute_precipitation_for_station(data, station, n_neighbors=5):
    station_data = data[data['station_id'] == station]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    station_data_imputed = imputer.fit_transform(station_data[['precipitation']])
    station_data['precipitation_imputed'] = station_data_imputed
    return station_data

def impute_precipitation_parallel(data, n_jobs=4):
    stations = data['station_id'].unique()
    pool = mp.Pool(processes=n_jobs)
    results = [
        pool.apply_async(impute_precipitation_for_station, args=(data, station,))
        for station in stations
    ]
    imputed_data = pd.concat([result.get() for result in results], ignore_index=True)
    pool.close()
    pool.join()
    return imputed_data

# Crear datos de ejemplo
data = pd.DataFrame({
    'station_id': [1, 1, 2, 2, 3, 3],
    'precipitation': [1, np.nan, 3, 4, np.nan, 6]
})

imputed_data = impute_precipitation_parallel(data)
print(imputed_data)