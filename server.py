import os
import sys
import json
import subprocess
import tempfile
import shutil
from flask import Flask, request, jsonify
from datetime import datetime
import traceback

app = Flask(__name__)

# Путь к основному скрипту (предполагается, что main.py лежит рядом)
MAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "main.py")

# Фиксированные параметры обработки (не передаются через запрос)
DEFAULT_GRID_SIZE = 50
DEFAULT_FORCE_DOWNLOAD = True


@app.route('/api/cloud-data', methods=['GET'])
def cloud_data():
    
    try:
        # Чтение параметров
        north = request.args.get('north', type=float)
        south = request.args.get('south', type=float)
        east = request.args.get('east', type=float)
        west = request.args.get('west', type=float)
        date = request.args.get('date')
        # source = request.args.get('source')  # пока не используется

        # Проверка обязательных параметров
        if None in (north, south, east, west, date):
            return jsonify({
                'error': 'Missing required parameters: north, south, east, west, date'
            }), 400

        # Проверка формата даты
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Invalid date format, expected YYYY-MM-DD'}), 400

        # Создаём временную директорию для выходных файлов
        temp_dir = tempfile.mkdtemp(prefix="earthdata_")
        output_json = os.path.join(temp_dir, "grid_output.json")

        # Формируем команду для запуска main.py
        cmd = [
            sys.executable,
            MAIN_SCRIPT,
            '--date', date,
            '--north', str(north),
            '--south', str(south),
            '--east', str(east),
            '--west', str(west),
            '--grid-size', str(DEFAULT_GRID_SIZE),
            '--force-download', str(DEFAULT_FORCE_DOWNLOAD).lower(),
            '--output-json', output_json
        ]

        # Запускаем процесс и ждём завершения
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Проверяем код возврата
        if result.returncode != 0:
            error_msg = f"main.py завершился с ошибкой (код {result.returncode})\n"
            error_msg += f"STDERR:\n{result.stderr}"
            return jsonify({'error': error_msg}), 500

        # Читаем выходной JSON-файл
        if not os.path.exists(output_json):
            return jsonify({'error': f'Выходной файл {output_json} не создан'}), 500

        with open(output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Удаляем временную директорию
        shutil.rmtree(temp_dir, ignore_errors=True)

        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Запуск сервера (для отладки, в production используйте WSGI-сервер)
    app.run(host='0.0.0.0', port=5000, debug=False)