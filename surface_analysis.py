import os
import numpy as np
import earthaccess
from pyhdf.SD import SD, SDC
from utils import get_all_hdf_files, save_json

DATA_DIR = r"C:\Test"
OUTPUT_JSON = os.path.join(DATA_DIR, "grid_output.json")

# --------------------- СКАЧИВАНИЕ ---------------------

def download_product(short_name, bounds, date, count=1):
    bbox = (bounds['west'], bounds['south'], bounds['east'], bounds['north'])
    temporal = (date, date)
    try:
        results = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox,
            temporal=temporal,
            count=count
        )
    except Exception as e:
        print(f"Ошибка поиска для {short_name}: {e}")
        return []
    if not results:
        print(f"Данные {short_name} не найдены для {date}.")
        return []
    os.makedirs(DATA_DIR, exist_ok=True)
    downloaded = earthaccess.download(results, DATA_DIR)
    print(f"Скачано {len(downloaded)} файлов для {short_name}")
    for f in downloaded:
        print(f"  -> {f}")
    return downloaded

def find_file_by_date(product_prefix, date):
    from datetime import datetime
    dt = datetime.strptime(date, "%Y-%m-%d")
    julian = dt.strftime("%Y%j")
    pattern = f"{product_prefix}.A{julian}"
    all_files = get_all_hdf_files(DATA_DIR)
    for f in all_files:
        if pattern in f:
            return f
    return None

# --------------------- ЧТЕНИЕ ---------------------

def read_mod29_sea_ice(filepath):
    """Читает Sea_Ice и координаты из MOD29_L2."""
    f = SD(filepath, SDC.READ)
    try:
        ice = f.select('Sea_Ice').get()
        lat = f.select('Latitude').get()
        lon = f.select('Longitude').get()
        print("MOD29: загружен Sea_Ice и координаты.")
    except Exception as e:
        print(f"Ошибка чтения MOD29: {e}")
        f.end()
        return None, None, None
    f.end()
    return ice, lat, lon

def read_mod06_cloud_mask(filepath):
    f = SD(filepath, SDC.READ)
    try:
        cm = f.select('Cloud_Mask_1km').get()
        cloudy = (cm[:, :, 0] & 0b00000011) != 3
        lat = f.select('Latitude').get()
        lon = f.select('Longitude').get()
    except Exception:
        try:
            cm = f.select('Cloud_Mask_5km').get()
            cloudy = (cm[:, :, 0] & 0b00000011) != 3
            lat = f.select('Latitude').get()
            lon = f.select('Longitude').get()
        except Exception as e:
            print(f"Не удалось прочитать Cloud_Mask: {e}")
            f.end()
            return None, None, None
    f.end()
    return cloudy, lat, lon

# --------------------- РЕСЭМПЛИНГ ---------------------

def resample_to_grid(data, lat, lon, lat_grid, lon_grid):
    if data is None or lat is None or lon is None:
        return np.full((len(lat_grid), len(lon_grid)), np.nan)
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

# --------------------- КЛАССИФИКАЦИЯ ---------------------

def classify_surface(user_bounds, date, grid_size=50, force_download=False):
    # 1. Скачиваем продукты (только MOD29 и MOD06)
    if force_download:
        download_product("MOD29_L2", user_bounds, date, count=1)   # для льда
        download_product("MOD06_L2", user_bounds, date, count=1)   # для облаков

    # 2. Ищем файлы
    mod29_file = find_file_by_date("MOD29_L2", date)
    mod06_file = find_file_by_date("MOD06_L2", date)

    # 3. Читаем данные
    ice, lat29, lon29 = (None, None, None)
    if mod29_file:
        ice, lat29, lon29 = read_mod29_sea_ice(mod29_file)
        if ice is None:
            print("Ошибка чтения MOD29, но продолжим без льда.")
    else:
        print("MOD29 не найден, лёд не будет выделяться.")

    cloud_mask, lat06, lon06 = (None, None, None)
    if mod06_file:
        cloud_mask, lat06, lon06 = read_mod06_cloud_mask(mod06_file)
    else:
        print("MOD06 не найден, облака не будут выделяться.")

    # 4. Строим сетку
    north, south, east, west = user_bounds['north'], user_bounds['south'], user_bounds['east'], user_bounds['west']
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    lat_grid = np.array([south + (i + 0.5) * lat_step for i in range(grid_size)])
    lon_grid = np.array([west + (j + 0.5) * lon_step for j in range(grid_size)])

    # 5. Ресемплинг
    ice_grid = resample_to_grid(ice, lat29, lon29, lat_grid, lon_grid) if ice is not None else np.full((grid_size, grid_size), np.nan)
    cloud_grid = resample_to_grid(cloud_mask.astype(np.float32) if cloud_mask is not None else None,
                                  lat06, lon06, lat_grid, lon_grid) if cloud_mask is not None else np.full((grid_size, grid_size), np.nan)

    # 6. Классификация: облака > лёд > вода
    classification = np.full((grid_size, grid_size), -1, dtype=np.int8)
    for i in range(grid_size):
        for j in range(grid_size):
            # Облака
            if cloud_grid[i, j] > 0.5:
                classification[i, j] = 3
                continue

            # Лёд
            if not np.isnan(ice_grid[i, j]) and ice_grid[i, j] == 1:
                classification[i, j] = 1
                continue

            # Вода
            classification[i, j] = 0

    # 7. Сохраняем в JSON
    row_labels = [f"{south + (i+0.5)*lat_step:.2f}" for i in range(grid_size)]
    col_labels = [f"{west + (j+0.5)*lon_step:.2f}" for j in range(grid_size)]

    surface_dict = {
        "rows": grid_size,
        "cols": grid_size,
        "data": classification.tolist(),
        "rowLabels": row_labels,
        "colLabels": col_labels,
        "class_names": {
            "-1": "undefined",
            "0": "water",
            "1": "sea_ice",
            "3": "cloud"
        }
    }

    import json
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing['surface'] = surface_dict
    save_json(existing, OUTPUT_JSON)
    print(f"Поверхностная классификация добавлена в {OUTPUT_JSON}")
    return existing