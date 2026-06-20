import os
import earthaccess
from utils import get_file_bounds, get_all_hdf_files

DATA_DIR = r"C:\Test"

def download_hdf_for_bbox(bounds, date, count=5):
    bbox = (bounds['west'], bounds['south'], bounds['east'], bounds['north'])
    temporal = (date, date)
    try:
        results = earthaccess.search_data(
            short_name="MOD06_L2",
            bounding_box=bbox,
            temporal=temporal,
            count=count
        )
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return []
    if not results:
        print(f"Данные не найдены для заданного прямоугольника и даты {date}.")
        return []
    os.makedirs(DATA_DIR, exist_ok=True)
    downloaded = earthaccess.download(results, DATA_DIR)
    print(f"Скачано файлов: {len(downloaded)} для даты {date}")
    return downloaded

def find_suitable(files, user_bounds, date=None):
    suitable = []
    # Если дата указана, преобразуем в юлианский формат для фильтрации
    julian = None
    if date:
        from utils import date_to_julian
        julian = date_to_julian(date)
    for f in files:
        # Фильтр по дате, если задана
        if julian and f" A{julian}" not in f:  # ищем подстроку с точкой
            # В имени файла дата обычно после A, например A2026136
            # Проверим, содержит ли имя A{date}
            if f"A{julian}" not in f:
                continue
        bounds = get_file_bounds(f)
        if bounds is None:
            continue
        min_lat, max_lat, min_lon, max_lon = bounds
        if (min_lat <= user_bounds['north'] and max_lat >= user_bounds['south'] and
            min_lon <= user_bounds['east'] and max_lon >= user_bounds['west']):
            suitable.append(f)
    return suitable

def ensure_files_for_bbox(user_bounds, date, force_download=False, count_first=1, count_second=1):
    if force_download:
        print(f"Скачиваем данные за {date} (принудительно)...")
        download_hdf_for_bbox(user_bounds, date, count_first)
    else:
        hdf_files = get_all_hdf_files(DATA_DIR)
        suitable = find_suitable(hdf_files, user_bounds, date=date)
        if not suitable:
            print(f"Нет файлов, покрывающих область. Скачиваем за {date}...")
            download_hdf_for_bbox(user_bounds, date, count_first)
        else:
            print(f"Найдены подходящие файлы для {date}, скачивание не требуется.")
            return suitable

    hdf_files = get_all_hdf_files(DATA_DIR)
    suitable = find_suitable(hdf_files, user_bounds, date=date)
    if not suitable:
        print(f"После скачивания не найдены подходящие файлы для {date}. Пробуем дополнительную порцию...")
        download_hdf_for_bbox(user_bounds, date, count_second)
        hdf_files = get_all_hdf_files(DATA_DIR)
        suitable = find_suitable(hdf_files, user_bounds, date=date)
    return suitable