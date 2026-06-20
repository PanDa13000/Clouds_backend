import os
import earthaccess

DEFAULT_CREDENTIALS_FILE = "earthdata_credentials.txt"

def login_earthdata(credentials_file=None, interactive_fallback=False):
    """
    Выполняет вход в EarthData, используя учётные данные из файла или интерактивно.
    """
    if credentials_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        possible_paths = [
            os.path.join(script_dir, DEFAULT_CREDENTIALS_FILE),
            os.path.join(cwd, DEFAULT_CREDENTIALS_FILE),
            os.path.join(home, DEFAULT_CREDENTIALS_FILE)
        ]
        cred_path = next((p for p in possible_paths if os.path.isfile(p)), None)
        if cred_path is None:
            print("Файл учётных данных не найден ни в одной из папок:")
            for p in possible_paths:
                print(f"  {p}")
    else:
        cred_path = credentials_file if os.path.isfile(credentials_file) else None

    username = None
    password = None
    if cred_path:
        try:
            with open(cred_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    username = lines[0].strip()
                    password = lines[1].strip()
                else:
                    print(f"Файл {cred_path} должен содержать логин и пароль на отдельных строках.")
        except Exception as e:
            print(f"Ошибка чтения файла учётных данных: {e}")

    if username and password:
        # Устанавливаем переменные окружения для earthaccess
        os.environ['EARTHDATA_USERNAME'] = username
        os.environ['EARTHDATA_PASSWORD'] = password
        try:
            earthaccess.login(strategy="environment", persist=True)
            print("Вход в EarthData выполнен успешно (учётные данные из файла).")
            return True
        except Exception as e:
            print(f"Ошибка входа с учётными данными из файла: {e}")
            if interactive_fallback:
                print("Попытка интерактивного входа...")
                try:
                    earthaccess.login(strategy="interactive", persist=True)
                    print("Интерактивный вход выполнен успешно.")
                    return True
                except Exception as e2:
                    print(f"Интерактивный вход не удался: {e2}")
                    return False
            return False
    else:
        if interactive_fallback:
            print("Файл учётных данных не найден или некорректен. Попытка интерактивного входа...")
            try:
                earthaccess.login(strategy="interactive", persist=True)
                print("Интерактивный вход выполнен успешно.")
                return True
            except Exception as e:
                print(f"Интерактивный вход не удался: {e}")
                return False
        else:
            print("Учётные данные не найдены. Укажите файл или включите interactive_fallback.")
            return False