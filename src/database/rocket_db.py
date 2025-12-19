import sqlite3
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class RocketLaunch:
    """
    Опис одного пуску ракети-носія.

    Цю структуру можна використовувати в UML як елемент
    «класу-сутності» Launch у базі даних.
    """
    launch_date: str          # дата запуску у форматі 'YYYY-MM-DD'
    rocket_name: str          # назва ракети
    payload_mass: float       # маса корисного вантажу, кг
    orbit: str                # тип орбіти
    metric_value: float       # числовий параметр для часових рядів


class RocketLaunchDB:
    """
    Простий клас для роботи з БД запусків ракет на SQLite.

    Показує, як у Python можна:
    - підключити базу даних (sqlite3.connect),
    - створити таблиці,
    - додати записи,
    - побудувати часовий ряд (наприклад, кількість запусків по роках).
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # За замовчуванням зберігаємо БД в папці data/
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            self.db_path = str(data_dir / "rockets.db")
        else:
            self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Встановлення з'єднання з БД."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def create_tables(self):
        """Створення таблиці запусків, якщо вона ще не існує."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS launches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                launch_date TEXT NOT NULL,
                rocket_name TEXT NOT NULL,
                payload_mass REAL,
                orbit TEXT,
                metric_value REAL
            );
            """
        )
        conn.commit()

    def insert_launch(self, launch: RocketLaunch, check_duplicates: bool = True):
        """
        Додавання одного запису про пуск ракети.
        
        :param launch: об'єкт RocketLaunch для додавання
        :param check_duplicates: чи перевіряти на дублікати (за датою та назвою ракети)
        """
        conn = self.connect()
        cur = conn.cursor()
        
        # Перевірка на дублікати (якщо увімкнено)
        if check_duplicates:
            cur.execute(
                """
                SELECT COUNT(*) FROM launches 
                WHERE launch_date = ? AND rocket_name = ?;
                """,
                (launch.launch_date, launch.rocket_name)
            )
            if cur.fetchone()[0] > 0:
                return  # Дублікат знайдено, пропускаємо
        
        cur.execute(
            """
            INSERT INTO launches (launch_date, rocket_name, payload_mass, orbit, metric_value)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                launch.launch_date,
                launch.rocket_name,
                launch.payload_mass,
                launch.orbit,
                launch.metric_value,
            ),
        )
        conn.commit()

    def ensure_demo_data(self, force_reload: bool = False):
        """
        Заповнення БД тестовими даними, якщо вона порожня.

        Це зручно для навчальних прикладів: не потрібно шукати
        реальний CSV, але структура БД вже «космічна».
        
        :param force_reload: якщо True, перезавантажити дані з CSV навіть якщо БД не порожня
        """
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM launches;")
        count = cur.fetchone()[0]
        
        # Якщо дані вже є і не потрібно примусово перезавантажувати
        if count > 0 and not force_reload:
            print(f"База даних вже містить {count} записів. Пропускаємо завантаження.")
            return  # дані вже є

        # Спробуємо завантажити дані з CSV файлів, якщо вони є
        csv_files = [
            Path(__file__).parent / "launches_2025__01-11.csv",
            Path(__file__).parent / "launches_2025-11.csv",
        ]
        
        loaded_count = 0
        for csv_file in csv_files:
            if csv_file.exists():
                try:
                    # Завжди використовуємо check_duplicates=True, щоб уникнути дублікатів
                    self.load_from_csv(str(csv_file), clear_existing=False)
                    loaded_count += 1
                    print(f"Завантажено дані з {csv_file.name}")
                except Exception as e:
                    print(f"Помилка завантаження {csv_file.name}: {e}")

        # Перевіримо чи є дані після завантаження CSV
        cur.execute("SELECT COUNT(*) FROM launches;")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"Всього записів в БД: {count}")
            return  # дані завантажені з CSV

        # Якщо CSV немає, використовуємо демо-дані
        print("CSV файли не знайдено, використовуємо демо-дані")
        demo_launches = [
            RocketLaunch("2015-03-19", "Zenit-3SLB", 3500, "LEO", 1),
            RocketLaunch("2016-06-12", "Falcon 9", 5500, "GTO", 1),
            RocketLaunch("2017-09-07", "Soyuz-2", 2200, "LEO", 1),
            RocketLaunch("2018-02-06", "Falcon Heavy", 6380, "HEO", 1),
            RocketLaunch("2019-04-11", "Electron", 300, "LEO", 1),
            RocketLaunch("2020-05-30", "Falcon 9 Crew Dragon", 12000, "LEO", 1),
            RocketLaunch("2021-09-15", "Falcon 9 Inspiration4", 4000, "LEO", 1),
            RocketLaunch("2022-11-16", "SLS Artemis I", 27000, "Lunar", 1),
            RocketLaunch("2023-04-20", "Starship (test)", 100000, "Test", 1),
        ]

        for launch in demo_launches:
            self.insert_launch(launch, check_duplicates=False)

    def load_from_csv(self, csv_path: str, clear_existing: bool = False):
        """
        Завантаження даних про запуски ракет з CSV файлу в базу даних.

        :param csv_path: шлях до CSV файлу з даними про запуски
        :param clear_existing: чи очистити існуючі дані перед завантаженням
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не знайдено: {csv_path}")

        df = pd.read_csv(csv_path)

        # Перевірка наявності необхідних стовпців
        required_columns = ['launcher', 'date']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV файл повинен містити стовпці: {required_columns}")

        if clear_existing:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute("DELETE FROM launches;")
            conn.commit()

        # Обробка даних
        for _, row in df.iterrows():
            try:
                # Парсинг дати (може бути в різних форматах)
                date_str = str(row['date'])
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]  # Беремо тільки дату
                # Перевірка формату дати
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    # Спробуємо інші формати
                    try:
                        dt = pd.to_datetime(date_str)
                        date_str = dt.strftime('%Y-%m-%d')
                    except:
                        print(f"Невідомий формат дати: {date_str}, пропускаємо рядок")
                        continue
                
                # Парсинг маси корисного вантажу (якщо є)
                payload_mass = 0.0
                if 'payload_mass' in df.columns:
                    try:
                        payload_mass = float(row.get('payload_mass', 0.0))
                    except (ValueError, TypeError):
                        payload_mass = 0.0

                # Орбіта
                orbit = str(row.get('orbit', 'Unknown'))

                launch = RocketLaunch(
                    launch_date=date_str,
                    rocket_name=str(row['launcher']),
                    payload_mass=payload_mass,
                    orbit=orbit,
                    metric_value=1.0  # Для підрахунку кількості
                )
                self.insert_launch(launch, check_duplicates=True)
            except Exception as e:
                print(f"Помилка обробки рядка: {e}")
                continue

    def load_series(self, metric: str = "count_per_year") -> np.ndarray:
        """
        Побудова часового ряду з БД.

        :param metric:
            - 'count_per_year'  – кількість запусків по роках;
            - 'avg_payload_per_year' – середня маса корисного вантажу по роках;
            - 'count_per_month' – кількість запусків по місяцях (для прогнозу на грудень 2025);
            - 'count_per_month_2025' – кількість запусків по місяцях тільки за 2025 рік.
        :return: одномірний numpy-масив значень.
        """
        conn = self.connect()
        cur = conn.cursor()

        if metric == "avg_payload_per_year":
            query = """
                SELECT strftime('%Y', launch_date) AS y, AVG(payload_mass)
                FROM launches
                GROUP BY y
                ORDER BY y;
            """
        elif metric == "count_per_month":
            query = """
                SELECT strftime('%Y-%m', launch_date) AS ym, COUNT(*)
                FROM launches
                GROUP BY ym
                ORDER BY ym;
            """
        elif metric == "count_per_month_2025":
            query = """
                SELECT strftime('%m', launch_date) AS m, COUNT(*)
                FROM launches
                WHERE strftime('%Y', launch_date) = '2025'
                GROUP BY m
                ORDER BY m;
            """
        else:  # count_per_year за замовчуванням
            query = """
                SELECT strftime('%Y', launch_date) AS y, COUNT(*)
                FROM launches
                GROUP BY y
                ORDER BY y;
            """

        cur.execute(query)
        rows = cur.fetchall()
        # Повертаємо тільки числовий ряд; роки/місяці можна зберегти окремо при потребі.
        values = [row[1] for row in rows]
        return np.array(values, dtype=float)


__all__ = ["RocketLaunchDB", "RocketLaunch"]


