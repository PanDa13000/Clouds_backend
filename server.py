import os
import sys
import tempfile
import shutil
import json
import threading
from flask import Flask, request, jsonify
from datetime import datetime
import traceback

# Импорт ваших модулей
from auth_utils import login_earthdata
from download_manager import ensure_files_for_bbox
from hdf_processor import extract_data_from_hdf_files
from surface_analysis import classify_surface
from spectral_processor import extract_spectral_data
from utils import get_previous_dates

app = Flask(__name__)
process_lock = threading.Lock()

DEFAULT_GRID_SIZE = 50
FORCE_DOWNLOAD = True  # можно переопределить через параметр запроса

def build_analysis(main_data, daily_data, main_date, previous_dates):
    """Формирует раздел analysis (вердикт, динамика) по аналогии с main.py"""
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

    return {
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

def process_request(bounds, date, grid_size, force_download):
    """Основная функция обработки: скачивание, извлечение, классификация, спектральные данные"""
    # Создаём временную директорию для этого запроса
    temp_dir = tempfile.mkdtemp(prefix="earthdata_")
    print(f"Временная директория: {temp_dir}")

    # Сохраняем оригинальные глобальные переменные, чтобы восстановить потом
    import download_manager
    import hdf_processor
    import surface_analysis

    orig_download_dir = download_manager.DATA_DIR
    orig_hdf_dir = hdf_processor.DATA_DIR
    orig_hdf_output = hdf_processor.OUTPUT_JSON
    orig_surface_dir = surface_analysis.DATA_DIR
    orig_surface_output = surface_analysis.OUTPUT_JSON

    new_data_dir = temp_dir
    new_output_json = os.path.join(new_data_dir, "grid_output.json")

    # Переопределяем глобальные переменные
    download_manager.DATA_DIR = new_data_dir
    hdf_processor.DATA_DIR = new_data_dir
    hdf_processor.OUTPUT_JSON = new_output_json
    surface_analysis.DATA_DIR = new_data_dir
    surface_analysis.OUTPUT_JSON = new_output_json

    try:
        # Вход в EarthData (учётные данные из файла)
        if not login_earthdata(interactive_fallback=False):
            raise Exception("Не удалось войти в EarthData")

        # 1. Получаем файлы для основной даты
        main_hdf_files = ensure_files_for_bbox(bounds, date, force_download=force_download)
        if not main_hdf_files:
            raise Exception(f"Не удалось получить данные для даты {date}")
        main_hdf_file = main_hdf_files[0]
        print(f"Основной файл: {main_hdf_file}")

        # 2. Извлекаем данные из основной даты
        main_data = extract_data_from_hdf_files(bounds, grid_size, [main_hdf_file])
        if main_data is None:
            raise Exception("Ошибка извлечения данных из файла основной даты")

        # 3. Обработка предыдущих дат для динамики
        dates = get_previous_dates(date, n=3)
        previous_dates = [d for d in dates if d != date]
        daily_data = {date: main_data}
        for d in previous_dates:
            print(f"Обработка даты для динамики: {d}")
            hdf_files = ensure_files_for_bbox(bounds, d, force_download=force_download)
            if not hdf_files:
                daily_data[d] = None
                continue
            data = extract_data_from_hdf_files(bounds, grid_size, hdf_files)
            daily_data[d] = data

        # 4. Формируем анализ
        analysis = build_analysis(main_data, daily_data, date, previous_dates)

        # 5. Собираем основной результат
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
        result = {
            'status': 'data_available',
            'gridSize': grid_size,
            'bounds': bounds,
            'analysis': analysis,
            'cloudGrid': cloud_grid_dict,
            'temperatureGrid': temp_grid_dict
        }

        # 6. Классификация поверхности (сохраняет в тот же JSON)
        classify_surface(bounds, date, grid_size, force_download=force_download)
        if os.path.exists(new_output_json):
            with open(new_output_json, 'r') as f:
                temp_result = json.load(f)
            if 'surface' in temp_result:
                result['surface'] = temp_result['surface']

        # 7. Спектральные данные
        spectral_data = extract_spectral_data(bounds, grid_size, main_hdf_file)
        if spectral_data is not None:
            result['spectralGrids'] = spectral_data

        return result

    finally:
        # Восстанавливаем глобальные переменные
        download_manager.DATA_DIR = orig_download_dir
        hdf_processor.DATA_DIR = orig_hdf_dir
        hdf_processor.OUTPUT_JSON = orig_hdf_output
        surface_analysis.DATA_DIR = orig_surface_dir
        surface_analysis.OUTPUT_JSON = orig_surface_output

        # Удаляем временную директорию
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Временная директория {temp_dir} удалена")

@app.route('/api/cloud-data', methods=['GET'])
def cloud_data():
    """Endpoint для получения данных по облачности и температуре"""
    with process_lock:  # Блокировка, чтобы избежать конфликтов глобальных переменных
        try:
            # Чтение параметров
            north = request.args.get('north', type=float)
            south = request.args.get('south', type=float)
            east = request.args.get('east', type=float)
            west = request.args.get('west', type=float)
            date = request.args.get('date')
            grid_size = request.args.get('grid_size', DEFAULT_GRID_SIZE, type=int)
            force_download = request.args.get('force_download', 'true').lower() == 'true'

            if None in (north, south, east, west, date):
                return jsonify({'error': 'Missing required parameters: north, south, east, west, date'}), 400

            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return jsonify({'error': 'Invalid date format, expected YYYY-MM-DD'}), 400

            bounds = {'north': north, 'south': south, 'east': east, 'west': west}

            result = process_request(bounds, date, grid_size, force_download)
            return jsonify(result)

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Для отладки, но в production используйте gunicorn или просто через pm2 с flask
    app.run(host='0.0.0.0', port=5000)