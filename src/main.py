import sys
from pathlib import Path
from datetime import datetime

# Додаємо корінь проекту до sys.path для коректної роботи імпортів
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import argparse
import warnings

from src.ssa import SSA as LibrarySSA
from src.database import RocketLaunchDB

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'


def arima_analysis(time_series, order=(1, 1, 1), forecast_steps=10):
    """
    Допоміжна функція для побудови моделі ARIMA та отримання прогнозу.

    На діаграмі її зручно розглядати як «чорну скриньку»,
    що реалізує класичний статистичний підхід до прогнозування
    часових рядів для порівняння з SSA.
    """
    adf_result = adfuller(time_series)
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=forecast_steps)
    forecast_df = fitted_model.get_forecast(steps=forecast_steps)
    conf_int = forecast_df.conf_int()
    results = {
        'model': fitted_model,
        'forecast': forecast,
        'conf_int': conf_int,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'is_stationary': adf_result[1] < 0.05,
        'summary': fitted_model.summary()
    }
    return results


def generate_sample_data(n_points=200, seed=42):
    """
    Генерація штучного часового ряду.

    Це тестові дані для демонстрації роботи алгоритмів,
    які можна інтерпретувати як, наприклад, температуру або вологість.
    """
    np.random.seed(seed)
    t = np.arange(n_points)
    trend = 0.05 * t + 10
    seasonal1 = 5 * np.sin(2 * np.pi * t / 12)
    seasonal2 = 2 * np.sin(2 * np.pi * t / 4)
    noise = np.random.normal(0, 1, n_points)
    time_series = trend + seasonal1 + seasonal2 + noise
    return time_series, trend, seasonal1, seasonal2, noise


def load_csv_data(file_path, column=None):
    df = pd.read_csv(file_path)
    if column is None:
        column = df.columns[0]
    return df[column].values


def plot_ssa_results(ssa, original, components, title, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[0, 0].plot(original, 'b-', linewidth=1)
    axes[0, 0].set_title('Вихідний часовий ряд')
    axes[0, 0].set_xlabel('Час')
    axes[0, 0].set_ylabel('Значення')
    axes[0, 0].grid(True, alpha=0.3)
    
    contributions = ssa.get_contributions(20)
    axes[0, 1].bar(range(len(contributions)), contributions, color='steelblue')
    axes[0, 1].set_title('Внесок компонент (Singular Values)')
    axes[0, 1].set_xlabel('Номер компоненти')
    axes[0, 1].set_ylabel('Внесок (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    if len(components) > 0:
        axes[1, 0].plot(original, 'b-', alpha=0.5, label='Вихідний ряд')
        axes[1, 0].plot(components[0], 'r-', linewidth=2, label='Тренд')
        axes[1, 0].set_title('Тренд')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if len(components) > 1:
        axes[1, 1].plot(components[1], 'g-', linewidth=1, label='Періодика')
        axes[1, 1].set_title('Періодична компонента')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    if len(components) > 2:
        axes[2, 0].plot(components[2], 'purple', linewidth=0.5)
        axes[2, 0].set_title('Шум (залишкова компонента)')
        axes[2, 0].grid(True, alpha=0.3)
    
    if len(components) >= 2:
        reconstructed = components[0] + components[1]
        axes[2, 1].plot(original, 'b-', alpha=0.5, label='Вихідний ряд')
        axes[2, 1].plot(reconstructed, 'r-', linewidth=1.5, label='Реконструкція')
        axes[2, 1].set_title('Реконструйований ряд')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_arima_results(time_series, arima_results, forecast_steps, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    n = len(time_series)
    
    axes[0].plot(range(n), time_series, 'b-', label='Вихідний ряд')
    axes[0].plot(range(n, n + forecast_steps), arima_results['forecast'], 
                 'r-', linewidth=2, label='Прогноз ARIMA')
    conf_int = arima_results['conf_int']
    if hasattr(conf_int, 'iloc'):
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
    else:
        lower = conf_int[:, 0]
        upper = conf_int[:, 1]
    axes[0].fill_between(range(n, n + forecast_steps),
                         lower, upper,
                         color='red', alpha=0.2, label='95% довірчий інтервал')
    axes[0].set_title('ARIMA: Прогнозування часового ряду')
    axes[0].set_xlabel('Час')
    axes[0].set_ylabel('Значення')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    residuals = arima_results['model'].resid
    axes[1].plot(residuals, 'g-', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Залишки моделі ARIMA')
    axes[1].set_xlabel('Час')
    axes[1].set_ylabel('Залишок')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison(original, ssa_reconstructed, arima_forecast, forecast_steps, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(original)
    
    ax.plot(range(n), original, 'b-', alpha=0.7, label='Вихідний ряд', linewidth=1)
    ax.plot(range(n), ssa_reconstructed, 'g-', label='SSA реконструкція', linewidth=1.5)
    ax.plot(range(n, n + forecast_steps), arima_forecast, 'r--', 
            label='ARIMA прогноз', linewidth=2)
    ax.axvline(x=n-1, color='gray', linestyle=':', label='Початок прогнозу')
    ax.set_title('Порівняння методів: SSA (Гусениця) vs ARIMA', fontsize=12)
    ax.set_xlabel('Час')
    ax.set_ylabel('Значення')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='SSA (Гусениця) та ARIMA аналіз часових рядів',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', type=str, default=None,
                        help='Шлях до CSV файлу з даними')
    parser.add_argument('--column', type=str, default=None,
                        help='Назва стовпця з даними в CSV')
    parser.add_argument('--points', type=int, default=200,
                        help='Кількість точок для генерації тестових даних (за замовчуванням: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для генератора випадкових чисел (за замовчуванням: 42)')
    parser.add_argument('--window', type=int, default=50,
                        help='Довжина вікна L для SSA (за замовчуванням: 50)')
    parser.add_argument('--arima-p', type=int, default=2,
                        help='Параметр p для ARIMA (за замовчуванням: 2)')
    parser.add_argument('--arima-d', type=int, default=1,
                        help='Параметр d для ARIMA (за замовчуванням: 1)')
    parser.add_argument('--arima-q', type=int, default=2,
                        help='Параметр q для ARIMA (за замовчуванням: 2)')
    parser.add_argument('--forecast', type=int, default=20,
                        help='Кількість кроків прогнозу (за замовчуванням: 20)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директорія для збереження графіків та результатів (створюється автоматично)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Не показувати графіки (тільки зберегти)')
    parser.add_argument('--use-db', action='store_true',
                        help='Використовувати SQLite БД запусків ракет замість CSV/штучних даних')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Шлях до файлу SQLite БД (за замовчуванням: data/rockets.db)')
    parser.add_argument('--db-metric', type=str, default='count_per_year',
                        help='Метрика для часових рядів з БД: count_per_year, avg_payload_per_year, count_per_month, count_per_month_2025')
    parser.add_argument('--forecast-december-2025', action='store_true',
                        help='Спрогнозувати дані по пусках ракет на грудень 2025 року')
    
    return parser.parse_args()


class TimeSeriesDataset:
    """
    Клас «Набір часових даних».

    На UML-діаграмі це окремий блок «Джерело даних». Він відповідає
    за завантаження та зберігання часового ряду, але не виконує
    жодних розрахунків.
    """

    def __init__(self, values, name="Без назви", units=None, source=None):
        """
        :param values: одномірний numpy-масив або список значень
        :param name: ім'я набору (наприклад, «Температура повітря»)
        :param units: одиниці вимірювання (наприклад, «°C»)
        :param source: опис джерела («CSV файл», «БД», «тестові дані»)
        """
        self.values = np.array(values)
        self.name = name
        self.units = units
        self.source = source

    @classmethod
    def from_csv(cls, file_path, column=None, name=None, units=None):
        """
        Створення набору даних з CSV-файлу.

        Цей метод можна зв'язати на діаграмі з блоком «Підключення БД /
        зовнішніх файлів». Поки що використовуємо простий варіант – CSV.
        """
        data = load_csv_data(file_path, column)
        if name is None:
            name = f"Дані з файлу {file_path}"
        return cls(data, name=name, units=units, source="CSV файл")

    @classmethod
    def from_weather_example(cls, n_points=200, seed=42):
        """
        Приклад тестового набору даних «Погода» (температура/вологість).

        Надалі в звіті можна показати цей набір як демонстраційний
        для тестування алгоритму.
        """
        series, *_ = generate_sample_data(n_points=n_points, seed=seed)
        return cls(series,
                   name="Тестовий часовий ряд (погода)",
                   units="ум. од.",
                   source="Згенеровані дані")


class SSAAnalyzer:
    """
    Клас-виконавець SSA-аналізу для заданого набору даних.

    В архітектурі програми він є обчислювальним блоком, який
    отримує на вхід TimeSeriesDataset і повертає компоненти:
    тренд, періодика, шум.
    """

    def __init__(self, dataset: TimeSeriesDataset, window_length: int):
        self.dataset = dataset
        self.window_length = window_length
        # Використовуємо бібліотечну реалізацію SSA з модуля ssa_caterpillar
        self.ssa = LibrarySSA(dataset.values, window_length=window_length)
        self.components = None
        self.groups = None
        self.contributions = None

    def analyze(self):
        """
        Запускає повний цикл SSA:
        1) вкладення,
        2) SVD,
        3) групування,
        4) реконструкція.
        """
        self.ssa.embed()
        self.ssa.decompose()
        self.contributions = self.ssa.get_contributions(10)

        # Типова схема групування:
        #   перша компонента – тренд,
        #   друга і третя – періодичність,
        #   інші – шум.
        self.groups = [[0], [1, 2], list(range(3, 15))]
        self.components = self.ssa.reconstruct(self.groups)
        return self.components

    def plot(self, save_path):
        """
        Побудова фігури з основними результатами SSA.

        На UML-діаграмі це можна показати як вихідний інтерфейс
        для модуля візуалізації.
        """
        if self.components is None:
            self.analyze()
        plot_ssa_results(self.ssa, self.dataset.values,
                         self.components,
                         "SSA Аналіз (Метод Гусениця)",
                         save_path)
        return save_path


class ARIMAAnalyzer:
    """
    Клас для аналізу того самого часового ряду за допомогою ARIMA.

    Його зручно відобразити на UML як окремий обчислювальний блок,
    який конкурує/порівнюється з SSAAnalyzer.
    """

    def __init__(self, dataset: TimeSeriesDataset, order=(2, 1, 2), forecast_steps=20):
        self.dataset = dataset
        self.order = order
        self.forecast_steps = forecast_steps
        self.results = None

    def analyze(self):
        """Запускає оцінку моделі ARIMA та формує словник результатів."""
        self.results = arima_analysis(
            self.dataset.values,
            order=self.order,
            forecast_steps=self.forecast_steps
        )
        return self.results

    def plot(self, save_path):
        """
        Побудова графіків прогнозу ARIMA та залишків.
        """
        if self.results is None:
            self.analyze()
        plot_arima_results(self.dataset.values,
                           self.results,
                           self.forecast_steps,
                           save_path)
        return save_path


class ForecastPipeline:
    """
    Клас «Конвеєр прогнозування».

    На UML-діаграмі цей клас можна показати як головний керуючий блок:
    він отримує дані (TimeSeriesDataset), запускає обчислення
    (SSAAnalyzer, ARIMAAnalyzer) і передає результати модулю візуалізації.
    """

    def __init__(self, dataset: TimeSeriesDataset,
                 window_length: int,
                 arima_order=(2, 1, 2),
                 forecast_steps=20,
                 output_dir="results"):
        self.dataset = dataset
        self.window_length = window_length
        self.arima_order = arima_order
        self.forecast_steps = forecast_steps
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ssa_analyzer = SSAAnalyzer(dataset, window_length=window_length)
        self.arima_analyzer = ARIMAAnalyzer(dataset,
                                            order=arima_order,
                                            forecast_steps=forecast_steps)

    def run(self, show_plots=True, command_info=None):
        """
        Основний сценарій роботи програми:
        1) SSA-аналіз,
        2) ARIMA-аналіз,
        3) побудова та збереження всіх графіків,
        4) порівняння методів,
        5) збереження інформації про команду.
        """
        if show_plots is False:
            plt.ioff()

        # 1. SSA
        ssa_path = self.output_dir / "ssa_results.png"
        print(f"   Збереження SSA графіків: {ssa_path}")
        self.ssa_analyzer.analyze()
        self.ssa_analyzer.plot(ssa_path)

        # 2. ARIMA
        arima_path = self.output_dir / "arima_results.png"
        print(f"   Збереження ARIMA графіків: {arima_path}")
        self.arima_analyzer.analyze()
        self.arima_analyzer.plot(arima_path)

        # 3. Порівняння
        comparison_path = self.output_dir / "comparison.png"
        print(f"   Збереження порівняння: {comparison_path}")
        components = self.ssa_analyzer.components
        ssa_reconstructed = components[0] + components[1]
        plot_comparison(self.dataset.values,
                        ssa_reconstructed,
                        self.arima_analyzer.results['forecast'],
                        self.forecast_steps,
                        comparison_path)

        # 4. Збереження інформації про команду та параметри
        if command_info:
            command_path = self.output_dir / "command.txt"
            with open(command_path, 'w', encoding='utf-8') as f:
                f.write(f"Дата та час запуску: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\nВикористана команда:\n{command_info['command']}\n")
                f.write(f"\nПараметри:\n")
                for key, value in command_info['args'].items():
                    f.write(f"  {key}: {value}\n")
                f.write(f"\nДані:\n")
                f.write(f"  Назва набору: {self.dataset.name}\n")
                f.write(f"  Джерело: {self.dataset.source}\n")
                f.write(f"  Кількість точок: {len(self.dataset.values)}\n")
                f.write(f"  Довжина вікна SSA: {self.window_length}\n")
                f.write(f"  Параметри ARIMA: {self.arima_order}\n")
                f.write(f"  Кроків прогнозу: {self.forecast_steps}\n")
            print(f"   Збережено інформацію про команду: {command_path}")

        return {
            "ssa_analyzer": self.ssa_analyzer,
            "arima_analyzer": self.arima_analyzer,
            "paths": {
                "ssa": ssa_path,
                "arima": arima_path,
                "comparison": comparison_path
            }
        }


def main():
    args = parse_args()
    
    # Формуємо повну команду для збереження
    full_command = ' '.join(sys.argv)
    
    # Створюємо словник аргументів для збереження
    args_dict = {
        'data': args.data,
        'column': args.column,
        'points': args.points,
        'seed': args.seed,
        'window': args.window,
        'arima_p': args.arima_p,
        'arima_d': args.arima_d,
        'arima_q': args.arima_q,
        'forecast': args.forecast,
        'output_dir': args.output_dir,
        'use_db': args.use_db,
        'db_path': args.db_path,
        'db_metric': args.db_metric,
        'forecast_december_2025': args.forecast_december_2025,
    }
    
    print("=" * 70)
    print("СИНГУЛЯРНИЙ СПЕКТРАЛЬНИЙ АНАЛІЗ (SSA) - МЕТОД ГУСЕНИЦЯ (CATERPILLAR)")
    print("ТА ARIMA АНАЛІЗ ЧАСОВИХ РЯДІВ")
    print("=" * 70)
    print()
    
    # 1. Блок «Джерело даних»
    #    На UML-діаграмі це окремий прямокутник TimeSeriesDataset.
    print("1. ЗАВАНТАЖЕННЯ ДАНИХ")
    print("-" * 40)

    if args.use_db:
        print(f"   Використання бази даних: {args.db_path}")
        db = RocketLaunchDB(db_path=args.db_path)
        db.connect()
        db.create_tables()
        db.ensure_demo_data()
        
        # Якщо потрібен прогноз на грудень 2025, використовуємо дані за місяцями 2025 року
        if args.forecast_december_2025:
            metric = 'count_per_month_2025'
            dataset_name = "Кількість запусків ракет по місяцях 2025 року"
        else:
            metric = args.db_metric
            dataset_name = f"Дані про запуски ракет ({metric})"
        
        series = db.load_series(metric=metric)
        dataset = TimeSeriesDataset(
            series,
            name=dataset_name,
            units="кількість запусків",
            source=f"SQLite БД ({args.db_path})"
        )
        print(f"   Завантажено точок: {len(dataset.values)}")
        if args.forecast_december_2025:
            print(f"   Дані за місяцями 2025 року (січень-листопад)")
            print(f"   Прогнозуємо грудень 2025 року")
    elif args.data:
        print(f"   Завантаження з файлу: {args.data}")
        dataset = TimeSeriesDataset.from_csv(
            args.data,
            column=args.column,
            name="Користувацький часовий ряд",
            units="ум. од."
        )
        print(f"   Завантажено точок: {len(dataset.values)}")
    else:
        print(f"   Генерація тестових даних (приклад «погода»)...")
        dataset = TimeSeriesDataset.from_weather_example(
            n_points=args.points,
            seed=args.seed
        )
        print(f"   Згенеровано точок: {len(dataset.values)}")
        print(f"   Seed: {args.seed}")
    print()

    # 2. Налаштування обчислювальних модулів (SSA та ARIMA)
    print("2. НАЛАШТУВАННЯ ОБЧИСЛЮВАЛЬНИХ МОДУЛІВ")
    print("-" * 40)

    arima_order = (args.arima_p, args.arima_d, args.arima_q)

    print(f"   Модуль SSA:")
    print(f"   - Довжина вікна (L): {args.window}")
    print()
    print(f"   Модуль ARIMA{arima_order}:")
    print(f"   - p = {arima_order[0]} (авторегресія)")
    print(f"   - d = {arima_order[1]} (інтегрування)")
    print(f"   - q = {arima_order[2]} (ковзне середнє)")
    print(f"   - Горизонт прогнозу: {args.forecast} точок")
    print()

    # 3. Запуск конвеєра прогнозування
    print("3. ЗАПУСК КОНВЕЄРА ПРОГНОЗУВАННЯ")
    print("-" * 40)

    pipeline = ForecastPipeline(
        dataset=dataset,
        window_length=args.window,
        arima_order=arima_order,
        forecast_steps=args.forecast,
        output_dir=args.output_dir
    )

    # Підготовка інформації про команду для збереження
    command_info = {
        'command': full_command,
        'args': args_dict
    }

    results = pipeline.run(show_plots=not args.no_plots, command_info=command_info)

    # 4. Коротке текстове резюме результатів для звіту
    print()
    print("4. КОРОТКИЙ ОПИС РЕЗУЛЬТАТІВ")
    print("-" * 40)
    ssa_analyzer = results["ssa_analyzer"]
    arima_analyzer = results["arima_analyzer"]

    print("   Внесок перших 10 компонент SSA:")
    for i, c in enumerate(ssa_analyzer.contributions):
        print(f"   Компонента {i + 1}: {c:.2f}%")

    arima_results = arima_analyzer.results
    print()
    print(f"   Показники моделі ARIMA:")
    print(f"   - AIC: {arima_results['aic']:.2f}")
    print(f"   - BIC: {arima_results['bic']:.2f}")
    print(f"   - Тест Дікі-Фуллера p-value: {arima_results['adf_pvalue']:.4f}")
    print(f"   - Ряд стаціонарний: {'Так' if arima_results['is_stationary'] else 'Ні'}")

    # 5. Прогноз на грудень 2025 року (якщо потрібно)
    if args.forecast_december_2025:
        print()
        print("5. ПРОГНОЗ НА ГРУДЕНЬ 2025 РОКУ")
        print("-" * 40)
        
        # Отримуємо прогноз від ARIMA (останній крок - це прогноз на грудень)
        arima_forecast = arima_results['forecast']
        december_forecast = arima_forecast[-1] if len(arima_forecast) > 0 else None
        
        # Також спробуємо отримати прогноз від SSA
        ssa_forecast = ssa_analyzer.ssa.forecast(steps=1, use_components=[0, 1, 2])
        ssa_december_forecast = ssa_forecast[-1] if len(ssa_forecast) > 0 else None
        
        if december_forecast is not None:
            print(f"   Прогноз ARIMA на грудень 2025: {december_forecast:.2f} запусків")
        if ssa_december_forecast is not None:
            print(f"   Прогноз SSA на грудень 2025: {ssa_december_forecast:.2f} запусків")
        
        # Середнє значення прогнозів
        if december_forecast is not None and ssa_december_forecast is not None:
            avg_forecast = (december_forecast + ssa_december_forecast) / 2
            print(f"   Середній прогноз на грудень 2025: {avg_forecast:.2f} запусків")
        
        # Показуємо дані за попередні місяці для контексту
        if len(dataset.values) >= 11:
            print()
            print("   Дані за попередні місяці 2025 року:")
            months = ['Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень',
                     'Липень', 'Серпень', 'Вересень', 'Жовтень', 'Листопад']
            for i, (month, value) in enumerate(zip(months, dataset.values)):
                print(f"   {month}: {value:.0f} запусків")

    print()
    print("=" * 70)
    print("Аналіз завершено!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
