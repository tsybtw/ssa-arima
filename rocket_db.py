import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np


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

    def __init__(self, db_path: str = "rockets.db"):
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

    def insert_launch(self, launch: RocketLaunch):
        """Додавання одного запису про пуск ракети."""
        conn = self.connect()
        cur = conn.cursor()
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

    def ensure_demo_data(self):
        """
        Заповнення БД тестовими даними, якщо вона порожня.

        Це зручно для навчальних прикладів: не потрібно шукати
        реальний CSV, але структура БД вже «космічна».
        """
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM launches;")
        count = cur.fetchone()[0]
        if count > 0:
            return  # дані вже є

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
            self.insert_launch(launch)

    def load_series(self, metric: str = "count_per_year") -> np.ndarray:
        """
        Побудова часового ряду з БД.

        :param metric:
            - 'count_per_year'  – кількість запусків по роках;
            - 'avg_payload_per_year' – середня маса корисного вантажу по роках.
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
        else:  # count_per_year за замовчуванням
            query = """
                SELECT strftime('%Y', launch_date) AS y, COUNT(*)
                FROM launches
                GROUP BY y
                ORDER BY y;
            """

        cur.execute(query)
        rows = cur.fetchall()
        # Повертаємо тільки числовий ряд; роки можна зберегти окремо при потребі.
        values = [row[1] for row in rows]
        return np.array(values, dtype=float)


__all__ = ["RocketLaunchDB", "RocketLaunch"]


