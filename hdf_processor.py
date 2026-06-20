import os
import numpy as np
from pyhdf.SD import SD, SDC
from utils import save_json, clean_nan

DATA_DIR = r"C:\Test"
OUTPUT_JSON = os.path.join(DATA_DIR, "grid_output.json")

class HDFGridGenerator:
    
    def __init__(self, hdf_path):
        self.hdf_path = hdf_path
        self.lat = None
        self.lon = None
        self.data = {}
        self._load_data()

    def _load_data(self):
        f = SD(self.hdf_path, SDC.READ)
        try:
            self.lat = f.select('Latitude').get()
            self.lon = f.select('Longitude').get()
        except Exception:
            raise ValueError("Нет Latitude/Longitude")
        for name in ['Cloud_Fraction', 'Cloud_Top_Temperature']:
            try:
                var = f.select(name)
                raw = var.get()
                attrs = var.attributes()
                fill = attrs.get('_FillValue', None)
                valid_min = attrs.get('valid_min', None)
                valid_max = attrs.get('valid_max', None)
                scale = attrs.get('scale_factor', 1.0)
                offset = attrs.get('add_offset', 0.0)
                mask = np.ones_like(raw, dtype=bool)
                if fill is not None:
                    mask &= (raw != fill)
                if valid_min is not None:
                    mask &= (raw >= valid_min)
                if valid_max is not None:
                    mask &= (raw <= valid_max)
                physical = (raw.astype(np.float32) - offset) * scale
                physical[~mask] = np.nan
                if name == 'Cloud_Top_Temperature':
                    physical -= 273.15
                self.data[name] = physical
            except Exception:
                self.data[name] = None
        f.end()

    def generate_grid(self, bounds, grid_size=50, cloud_threshold=0.1):
        north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
        mask = (self.lat >= south) & (self.lat <= north) & (self.lon >= west) & (self.lon <= east)
        if not np.any(mask):
            return None

        lat_step = (north - south) / grid_size
        lon_step = (east - west) / grid_size
        cloud_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        temp_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

        cf = self.data.get('Cloud_Fraction')
        temp = self.data.get('Cloud_Top_Temperature')

        for i in range(grid_size):
            lat_min = south + i * lat_step
            lat_max = south + (i + 1) * lat_step
            for j in range(grid_size):
                lon_min = west + j * lon_step
                lon_max = west + (j + 1) * lon_step
                cell_mask = (self.lat >= lat_min) & (self.lat < lat_max) & \
                            (self.lon >= lon_min) & (self.lon < lon_max) & mask
                if not np.any(cell_mask):
                    continue

                if cf is not None and temp is not None:
                    cf_vals = cf[cell_mask]
                    temp_vals = temp[cell_mask]
                    good = ~np.isnan(cf_vals) & ~np.isnan(temp_vals)
                    if cloud_threshold is not None:
                        good &= (cf_vals > cloud_threshold)
                    if np.any(good):
                        cloud_grid[i, j] = np.mean(cf_vals[good])
                        temp_grid[i, j] = np.mean(temp_vals[good])
                elif cf is not None:
                    vals = cf[cell_mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        cloud_grid[i, j] = np.mean(vals)
                elif temp is not None:
                    vals = temp[cell_mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        temp_grid[i, j] = np.mean(vals)

        return {
            'cloudCover': cloud_grid.tolist(),
            'temperature': temp_grid.tolist()
        }

def process_hdf_files(user_bounds, grid_size, hdf_files):
    """
    Обрабатывает список файлов, строит сетку и сохраняет JSON.
    Возвращает путь к сохранённому файлу.
    """
    if not hdf_files:
        empty = {
            'gridSize': grid_size,
            'bounds': user_bounds,
            'analysis': {
                'cloudPercentage': None,
                'verdict': {'status': 'no_data', 'title': 'Нет данных', 'description': 'Не найдены файлы для обработки'},
                'temperature': {'max': None, 'avg': None},
                'dynamics': {'status': 'unknown', 'title': 'Нет данных', 'description': ''}
            },
            'cloudGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []},
            'temperatureGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []}
        }
        save_json(empty, OUTPUT_JSON)
        return OUTPUT_JSON

    # Пробуем первый файл
    for f in hdf_files:
        print(f"Попытка обработки файла: {f}")
        generator = HDFGridGenerator(f)
        grid_data = generator.generate_grid(user_bounds, grid_size)
        if grid_data is not None:
            break
    else:
        # ни один не подошёл
        empty = {
            'gridSize': grid_size,
            'bounds': user_bounds,
            'analysis': {
                'cloudPercentage': None,
                'verdict': {'status': 'no_data', 'title': 'Нет данных', 'description': 'Файлы не содержат данных в указанной области'},
                'temperature': {'max': None, 'avg': None},
                'dynamics': {'status': 'unknown', 'title': 'Нет данных', 'description': ''}
            },
            'cloudGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []},
            'temperatureGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []}
        }
        save_json(empty, OUTPUT_JSON)
        return OUTPUT_JSON

    # Получили данные
    cloud_mat = np.array(grid_data['cloudCover'])
    temp_mat = np.array(grid_data['temperature'])

    north = user_bounds['north']; south = user_bounds['south']; east = user_bounds['east']; west = user_bounds['west']
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    row_labels = [f"{south + (i+0.5)*lat_step:.2f}" for i in range(grid_size)]
    col_labels = [f"{west + (j+0.5)*lon_step:.2f}" for j in range(grid_size)]

    cloud_vals = cloud_mat[~np.isnan(cloud_mat)]
    temp_vals = temp_mat[~np.isnan(temp_mat)]

    cloud_percent = np.mean(cloud_vals) * 100 if len(cloud_vals) > 0 else None
    temp_max = np.max(temp_vals) if len(temp_vals) > 0 else None
    temp_avg = np.mean(temp_vals) if len(temp_vals) > 0 else None

    if cloud_percent is not None:
        if cloud_percent < 30:
            verdict_status, verdict_title, verdict_desc = 'good', 'Условия благоприятные', 'Облачность в пределах допустимого диапазона, видимость хорошая'
        elif cloud_percent < 60:
            verdict_status, verdict_title, verdict_desc = 'moderate', 'Умеренная облачность', 'Облачность средняя, возможны ограничения видимости'
        else:
            verdict_status, verdict_title, verdict_desc = 'bad', 'Высокая облачность', 'Облачность превышает норму, видимость ограничена'
    else:
        verdict_status, verdict_title, verdict_desc = 'unknown', 'Нет данных', 'Не удалось рассчитать облачность'

    dynamics = {'status': 'unknown', 'title': 'Нет данных', 'description': 'Динамика не вычисляется для одного снимка'}
    analysis = {
        'cloudPercentage': round(cloud_percent, 1) if cloud_percent is not None else None,
        'verdict': {'status': verdict_status, 'title': verdict_title, 'description': verdict_desc},
        'temperature': {'max': round(temp_max, 1) if temp_max is not None else None, 'avg': round(temp_avg, 1) if temp_avg is not None else None},
        'dynamics': dynamics
    }

    cloudGrid = {'rows': grid_size, 'cols': grid_size, 'data': cloud_mat.tolist(), 'rowLabels': row_labels, 'colLabels': col_labels}
    temperatureGrid = {'rows': grid_size, 'cols': grid_size, 'data': temp_mat.tolist(), 'rowLabels': row_labels, 'colLabels': col_labels}

    result = {'gridSize': grid_size, 'bounds': user_bounds, 'analysis': analysis, 'cloudGrid': cloudGrid, 'temperatureGrid': temperatureGrid}
    save_json(result, OUTPUT_JSON)
    print(f"Результат сохранён в {OUTPUT_JSON}")
    return OUTPUT_JSON

def extract_data_from_hdf_files(user_bounds, grid_size, hdf_files):
    """
    Обрабатывает список HDF-файлов и возвращает структуру данных:
    {
        'cloudGrid': <2D list>,
        'temperatureGrid': <2D list>,
        'cloudPercentage': float or None,
        'temp_max': float or None,
        'temp_avg': float or None,
        'rowLabels': list,
        'colLabels': list
    }
    Если файлы не подходят, возвращает None.
    """
    if not hdf_files:
        return None

    # Пробуем найти подходящий файл
    for f in hdf_files:
        print(f"Попытка обработки файла для извлечения: {f}")
        generator = HDFGridGenerator(f)
        grid_data = generator.generate_grid(user_bounds, grid_size)
        if grid_data is not None:
            break
    else:
        return None  # ни один файл не подошёл

    cloud_mat = np.array(grid_data['cloudCover'])
    temp_mat = np.array(grid_data['temperature'])

    north = user_bounds['north']
    south = user_bounds['south']
    east = user_bounds['east']
    west = user_bounds['west']
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    row_labels = [f"{south + (i + 0.5) * lat_step:.2f}" for i in range(grid_size)]
    col_labels = [f"{west + (j + 0.5) * lon_step:.2f}" for j in range(grid_size)]

    cloud_vals = cloud_mat[~np.isnan(cloud_mat)]
    temp_vals = temp_mat[~np.isnan(temp_mat)]

    cloud_percent = np.mean(cloud_vals) * 100 if len(cloud_vals) > 0 else None
    temp_max = np.max(temp_vals) if len(temp_vals) > 0 else None
    temp_avg = np.mean(temp_vals) if len(temp_vals) > 0 else None

    return {
        'cloudGrid': cloud_mat.tolist(),
        'temperatureGrid': temp_mat.tolist(),
        'cloudPercentage': cloud_percent,
        'temp_max': temp_max,
        'temp_avg': temp_avg,
        'rowLabels': row_labels,
        'colLabels': col_labels
    }