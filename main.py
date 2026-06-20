from download_manager import ensure_files_for_bbox
from hdf_processor import extract_data_from_hdf_files, save_json
from auth_utils import login_earthdata
from surface_analysis import classify_surface
from spectral_processor import extract_spectral_data
from utils import get_previous_dates
import argparse
import sys
import os
import numpy as np

DATA_DIR = r"C:\Test"
OUTPUT_JSON = os.path.join(DATA_DIR, "grid_output.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--north', type=float, required=True)
    parser.add_argument('--south', type=float, required=True)
    parser.add_argument('--east', type=float, required=True)
    parser.add_argument('--west', type=float, required=True)
    parser.add_argument('--grid-size', type=int, default=50)
    parser.add_argument('--force-download', type=bool, default=True)
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    # Вход в EarthData
    if not login_earthdata(interactive_fallback=True):
        print("Не удалось войти в EarthData. Выход.")
        sys.exit(1)

    user_bbox = {
        "north": args.north,
        "south": args.south,
        "east": args.east,
        "west": args.west
    }
    main_date = args.date
    grid_size = args.grid_size
    force_download = args.force_download

    # Переопределяем глобальные пути, чтобы все модули писали в одну папку
    # Для этого создадим папку для данных рядом с выходным JSON (или можно использовать общую)
    data_dir = os.path.dirname(args.output_json)
    os.makedirs(data_dir, exist_ok=True)

    # Переопределяем глобальные переменные в модулях
    import download_manager
    import hdf_processor
    import surface_analysis

    download_manager.DATA_DIR = data_dir
    hdf_processor.DATA_DIR = data_dir
    hdf_processor.OUTPUT_JSON = args.output_json
    surface_analysis.DATA_DIR = data_dir
    surface_analysis.OUTPUT_JSON = args.output_json

    # 1. Скачиваем/находим данные для основной даты
    print(f"\nОбработка основной даты: {main_date}")
    main_hdf_files = ensure_files_for_bbox(user_bbox, main_date, force_download=force_download)
    if not main_hdf_files:
        print(f"Не удалось получить данные для основной даты {main_date}.")
        empty_result = {
            'status': 'no_data',
            'message': f'Не удалось получить данные для даты {main_date} в указанной области.',
            'gridSize': grid_size,
            'bounds': user_bbox,
            'analysis': {
                'cloudPercentage': None,
                'verdict': {'status': 'no_data', 'title': 'Нет данных', 'description': 'Не удалось получить данные для указанной даты'},
                'temperature': {'max': None, 'avg': None},
                'dynamics': {'status': 'unknown', 'title': 'Нет данных', 'description': ''}
            },
            'cloudGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []},
            'temperatureGrid': {'rows': grid_size, 'cols': grid_size, 'data': [[None]*grid_size for _ in range(grid_size)], 'rowLabels': [], 'colLabels': []}
        }
        save_json(empty_result, OUTPUT_JSON)
        print(f"Пустой JSON сохранён в {OUTPUT_JSON}")
        sys.exit(0)

    # Берём первый подходящий файл для основной даты
    main_hdf_file = main_hdf_files[0]
    print(f"Найден файл для основной даты: {main_hdf_file}")

    # Извлекаем данные из файла основной даты
    main_data = extract_data_from_hdf_files(user_bbox, grid_size, [main_hdf_file])
    if main_data is None:
        print("Ошибка извлечения данных из файла основной даты.")
        sys.exit(1)

    # 2. Скачиваем и обрабатываем предыдущие даты для динамики
    dates = get_previous_dates(main_date, n=3)  # основная + 3 предыдущих
    previous_dates = [d for d in dates if d != main_date]

    daily_data = {main_date: main_data}
    for date in previous_dates:
        print(f"\nОбработка даты для динамики: {date}")
        hdf_files = ensure_files_for_bbox(user_bbox, date, force_download=force_download)
        if not hdf_files:
            print(f"Для даты {date} файлы не найдены.")
            daily_data[date] = None
            continue
        data = extract_data_from_hdf_files(user_bbox, grid_size, hdf_files)
        if data is None:
            print(f"Для даты {date} не удалось извлечь данные.")
            daily_data[date] = None
        else:
            daily_data[date] = data
            print(f"Для даты {date} данные получены. Облачность: {data['cloudPercentage']:.1f}%")

    # 3. Формируем вердикт и динамику
    cloud_pct = main_data['cloudPercentage']
    if cloud_pct is not None:
        if cloud_pct < 30:
            verdict_status, verdict_title, verdict_desc = 'good', 'Условия благоприятные', 'Облачность в пределах допустимого диапазона, видимость хорошая'
        elif cloud_pct < 60:
            verdict_status, verdict_title, verdict_desc = 'moderate', 'Умеренная облачность', 'Облачность средняя, возможны ограничения видимости'
        else:
            verdict_status, verdict_title, verdict_desc = 'bad', 'Высокая облачность', 'Облачность превышает норму, видимость ограничена'
    else:
        verdict_status, verdict_title, verdict_desc = 'unknown', 'Нет данных', 'Не удалось рассчитать облачность'

    # Динамика
    dates_with_data = [d for d in [main_date] + previous_dates if daily_data.get(d) is not None]
    cloud_series = []
    temp_series = []
    for d in dates_with_data:
        data = daily_data[d]
        cloud_series.append(data['cloudPercentage'] if data['cloudPercentage'] is not None else None)
        temp_series.append(data['temp_avg'] if data['temp_avg'] is not None else None)

    cloud_vals = [v for v in cloud_series if v is not None]
    temp_vals = [v for v in temp_series if v is not None]

    dynamics_status = 'unknown'
    dynamics_title = 'Нет данных'
    dynamics_description = 'Недостаточно данных для оценки динамики'

    if len(cloud_vals) >= 2 and len(temp_vals) >= 2:
        cloud_change = cloud_vals[-1] - cloud_vals[0]
        temp_change = temp_vals[-1] - temp_vals[0]
        cloud_trend = 'увеличивается' if cloud_change > 5 else 'уменьшается' if cloud_change < -5 else 'стабильна'
        temp_trend = 'растёт' if temp_change > 1.0 else 'падает' if temp_change < -1.0 else 'стабильна'
        dynamics_status = 'stable'
        if cloud_change > 5 and temp_change < -1:
            dynamics_status = 'deteriorating'
            dynamics_title = 'Ухудшение условий'
            dynamics_description = f'Облачность увеличилась на {cloud_change:.1f}%, температура понизилась на {abs(temp_change):.1f}°C'
        elif cloud_change < -5 and temp_change > 1:
            dynamics_status = 'improving'
            dynamics_title = 'Улучшение условий'
            dynamics_description = f'Облачность уменьшилась на {abs(cloud_change):.1f}%, температура повысилась на {temp_change:.1f}°C'
        else:
            dynamics_title = 'Условия стабильны'
            dynamics_description = f'За период облачность {cloud_trend} (изменение {cloud_change:+.1f}%), температура {temp_trend} (изменение {temp_change:+.1f}°C)'
    else:
        dynamics_description = 'Недостаточно временных данных для расчёта динамики'

    # 4. Собираем итоговый JSON
    cloud_grid_dict = {
        'rows': grid_size,
        'cols': grid_size,
        'data': main_data['cloudGrid'],
        'rowLabels': main_data['rowLabels'],
        'colLabels': main_data['colLabels']
    }
    temp_grid_dict = {
        'rows': grid_size,
        'cols': grid_size,
        'data': main_data['temperatureGrid'],
        'rowLabels': main_data['rowLabels'],
        'colLabels': main_data['colLabels']
    }

    analysis = {
        'cloudPercentage': round(cloud_pct, 1) if cloud_pct is not None else None,
        'verdict': {'status': verdict_status, 'title': verdict_title, 'description': verdict_desc},
        'temperature': {
            'max': round(main_data['temp_max'], 1) if main_data['temp_max'] is not None else None,
            'avg': round(main_data['temp_avg'], 1) if main_data['temp_avg'] is not None else None
        },
        'dynamics': {
            'status': dynamics_status,
            'title': dynamics_title,
            'description': dynamics_description
        }
    }

    result = {
        'status': 'data_available',
        'gridSize': grid_size,
        'bounds': user_bbox,
        'analysis': analysis,
        'cloudGrid': cloud_grid_dict,
        'temperatureGrid': temp_grid_dict
    }

    save_json(result, OUTPUT_JSON)
    print(f"Результат сохранён в {OUTPUT_JSON}")

    # 5. Классификация поверхности (используем файл основной даты)
    classify_surface(user_bbox, main_date, grid_size, force_download=False)  # не скачиваем повторно

    # 6. Спектральные данные (из файла основной даты)
    spectral_data = extract_spectral_data(user_bbox, grid_size, main_hdf_file)
    if spectral_data is not None:
        import json
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            current = json.load(f)
        current['spectralGrids'] = spectral_data
        save_json(current, OUTPUT_JSON)
        print("Спектральные данные добавлены в grid_output.json")
    else:
        print("Не удалось извлечь спектральные данные.")