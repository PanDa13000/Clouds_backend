import os
import numpy as np
from pyhdf.SD import SD, SDC
from utils import save_json

def resample_to_grid(data, lat, lon, lat_grid, lon_grid):
    """
    Ресемплирует данные на регулярную сетку (ближайший сосед).
    При несовпадении размеров обрезает до минимальной общей размерности.
    """
    if data is None or lat is None or lon is None:
        return np.full((len(lat_grid), len(lon_grid)), np.nan)
    if data.ndim != 2:
        print(f"Ожидались 2D данные, получена размерность {data.ndim}")
        return np.full((len(lat_grid), len(lon_grid)), np.nan)
    # Обрезаем до минимальной общей размерности
    min_rows = min(data.shape[0], lat.shape[0], lon.shape[0])
    min_cols = min(data.shape[1], lat.shape[1], lon.shape[1])
    data = data[:min_rows, :min_cols]
    lat = lat[:min_rows, :min_cols]
    lon = lon[:min_rows, :min_cols]

    nlat = len(lat_grid)
    nlon = len(lon_grid)
    result = np.full((nlat, nlon), np.nan, dtype=np.float32)
    lat_step = lat_grid[1] - lat_grid[0] if len(lat_grid) > 1 else 0.01
    lon_step = lon_grid[1] - lon_grid[0] if len(lon_grid) > 1 else 0.01
    lat_min = lat_grid[0] - lat_step/2
    lon_min = lon_grid[0] - lon_step/2

    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(lat[i, j]) or np.isnan(lon[i, j]):
                continue
            ilat = int((lat[i, j] - lat_min) / lat_step)
            ilon = int((lon[i, j] - lon_min) / lon_step)
            if 0 <= ilat < nlat and 0 <= ilon < nlon:
                result[ilat, ilon] = data[i, j]
    return result

def extract_spectral_data(user_bounds, grid_size, hdf_file):
    """
    Извлекает атмосферно скорректированную отражательную способность (Atm_Corr_Refl)
    из HDF-файла и ресемплирует на сетку grid_size x grid_size.
    Возвращает словарь с ключами:
        'bands' : список названий каналов,
        'data'  : список матриц (каждая размером grid_size x grid_size),
        'rows', 'cols', 'rowLabels', 'colLabels' – как в других сетках.
    Если данные отсутствуют или файл не найден, возвращает None.
    """
    if not hdf_file or not os.path.isfile(hdf_file):
        print(f"Файл не найден: {hdf_file}")
        return None

    try:
        f = SD(hdf_file, SDC.READ)
        lat = f.select('Latitude').get()
        lon = f.select('Longitude').get()
        # Пытаемся получить Atm_Corr_Refl
        try:
            sds = f.select('Atm_Corr_Refl')
        except Exception as e:
            print(f"В файле нет Atm_Corr_Refl (вероятно, ночной снимок): {e}")
            f.end()
            # Возвращаем структуру с NaN
            return _create_empty_spectral(user_bounds, grid_size)
        raw = sds.get()                     # форма (rows, cols, 6)
        attrs = sds.attributes()
        fill = attrs.get('_FillValue', -9999)
        scale = attrs.get('scale_factor', 1.0)
        offset = attrs.get('add_offset', 0.0)
        f.end()
    except Exception as e:
        print(f"Ошибка чтения спектральных данных: {e}")
        return _create_empty_spectral(user_bounds, grid_size)

    # Приводим к float32, заменяем fill на NaN, применяем масштаб
    data = raw.astype(np.float32)
    data[data == fill] = np.nan
    data = data * scale + offset

    # Проверяем, есть ли валидные значения
    if np.all(np.isnan(data)):
        print("Все значения в Atm_Corr_Refl равны NaN (ночной снимок).")
        return _create_empty_spectral(user_bounds, grid_size)

    # Строим сетку
    north, south, east, west = user_bounds['north'], user_bounds['south'], user_bounds['east'], user_bounds['west']
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    lat_grid = np.array([south + (i + 0.5) * lat_step for i in range(grid_size)])
    lon_grid = np.array([west + (j + 0.5) * lon_step for j in range(grid_size)])

    # Ресемплируем каждый канал
    band_data = []
    rows, cols, bands = data.shape
    for b in range(bands):
        band_matrix = resample_to_grid(data[:, :, b], lat, lon, lat_grid, lon_grid)
        band_data.append(band_matrix.tolist())

    # Имена каналов
    band_names = ["0.65um", "0.86um", "1.2um", "1.6um", "2.1um", "3.7um"]

    return {
        'bands': band_names,
        'data': band_data,
        'rows': grid_size,
        'cols': grid_size,
        'rowLabels': [f"{south + (i + 0.5) * lat_step:.2f}" for i in range(grid_size)],
        'colLabels': [f"{west + (j + 0.5) * lon_step:.2f}" for j in range(grid_size)]
    }

def _create_empty_spectral(user_bounds, grid_size):
    """Создаёт структуру с пустыми (NaN) спектральными данными."""
    north, south, east, west = user_bounds['north'], user_bounds['south'], user_bounds['east'], user_bounds['west']
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    band_names = ["0.65um", "0.86um", "1.2um", "1.6um", "2.1um", "3.7um"]
    empty_matrix = [[None] * grid_size for _ in range(grid_size)]
    band_data = [empty_matrix for _ in range(6)]
    return {
        'bands': band_names,
        'data': band_data,
        'rows': grid_size,
        'cols': grid_size,
        'rowLabels': [f"{south + (i + 0.5) * lat_step:.2f}" for i in range(grid_size)],
        'colLabels': [f"{west + (j + 0.5) * lon_step:.2f}" for j in range(grid_size)]
    }

def add_spectral_to_json(output_json_path, spectral_data):
    """
    Добавляет спектральные данные в существующий JSON-файл (в поле 'spectralGrids').
    Если поле уже существует, перезаписывает его.
    """
    if spectral_data is None:
        print("Нет спектральных данных для добавления.")
        return False

    import json
    if not os.path.exists(output_json_path):
        print(f"Файл {output_json_path} не найден. Создаём новый.")
        result = {}
    else:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

    result['spectralGrids'] = spectral_data
    save_json(result, output_json_path)
    print(f"Спектральные данные добавлены в {output_json_path}")
    return True