import os
import json
import math
import numpy as np
from pyhdf.SD import SD, SDC

def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj):
            return None
        return float(obj)
    else:
        return obj

def save_json(data, path):
    cleaned = clean_nan(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

def get_file_bounds(hdf_path):
    try:
        f = SD(hdf_path, SDC.READ)
        lat = f.select('Latitude').get()
        lon = f.select('Longitude').get()
        f.end()
        lat_valid = lat[~np.isnan(lat)]
        lon_valid = lon[~np.isnan(lon)]
        if lat_valid.size == 0 or lon_valid.size == 0:
            return None
        return (np.min(lat_valid), np.max(lat_valid), np.min(lon_valid), np.max(lon_valid))
    except Exception:
        return None

def get_all_hdf_files(directory):
    import glob
    return (glob.glob(os.path.join(directory, "*.hdf")) +
            glob.glob(os.path.join(directory, "*.HDF")) +
            glob.glob(os.path.join(directory, "*.h5")))

from datetime import datetime, timedelta

def get_previous_dates(date_str, n=3):
    """
    Возвращает список дат в формате YYYY-MM-DD: [date_str, date_str-1, ..., date_str-n]
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    return [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n + 1)]

def date_to_julian(date_str):
    """Преобразует дату в формате YYYY-MM-DD в юлианский день (YYYYDDD)."""
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%j")
