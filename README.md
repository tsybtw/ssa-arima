# SSA та ARIMA Аналіз Часових Рядів

Програма для аналізу часових рядів методами **SSA (Singular Spectrum Analysis)** — також відомий як метод **"Гусениця" (Caterpillar)** — та **ARIMA**.

## Встановлення залежностей

```bash
pip install numpy pandas matplotlib scipy statsmodels
```

або

```bash
pip install -r requirements.txt
```

## Запуск

### Базовий запуск (з тестовими даними)

```bash
python main.py
```

### Запуск з власними даними

```bash
python main.py --data my_data.csv --column price
```

## Параметри командного рядка

| Параметр | Тип | За замовчуванням | Опис |
|----------|-----|------------------|------|
| `--data` | str | None | Шлях до CSV файлу з даними |
| `--column` | str | None | Назва стовпця з даними в CSV (якщо не вказано — перший стовпець) |
| `--points` | int | 200 | Кількість точок для генерації тестових даних |
| `--seed` | int | 42 | Seed для генератора випадкових чисел |
| `--window` | int | 50 | Довжина вікна L для SSA аналізу |
| `--arima-p` | int | 2 | Параметр p для ARIMA (порядок авторегресії) |
| `--arima-d` | int | 1 | Параметр d для ARIMA (порядок інтегрування) |
| `--arima-q` | int | 2 | Параметр q для ARIMA (порядок ковзного середнього) |
| `--forecast` | int | 20 | Кількість кроків прогнозу |
| `--output-dir` | str | . | Директорія для збереження графіків |
| `--no-plots` | flag | False | Не показувати графіки (тільки зберегти) |
| `--use-db` | flag | False | Використати базу даних SQLite замість CSV/тестових даних |
| `--db-path` | str | rockets.db | Шлях до файлу бази даних SQLite |
| `--db-metric` | str | count_per_year | Метрика для побудови ряду: `count_per_year` або `avg_payload_per_year` |

## Приклади використання

### Генерація 500 точок з іншим seed

```bash
python main.py --points 500 --seed 123
```

### SSA з більшим вікном

```bash
python main.py --window 100
```

### ARIMA(1,1,1) з прогнозом на 50 кроків

```bash
python main.py --arima-p 1 --arima-d 1 --arima-q 1 --forecast 50
```

### Збереження графіків в окрему папку без показу

```bash
python main.py --output-dir ./results --no-plots
```

### Повний приклад з власними даними

```bash
python main.py --data sales.csv --column revenue --window 30 --arima-p 2 --arima-d 1 --arima-q 1 --forecast 12 --output-dir ./output
```

### Використання бази даних запусків ракет

```bash
python main.py --use-db --db-path rockets.db --db-metric count_per_year --window 30 --forecast 20
```

## Архітектура програми

Програма побудована за модульним принципом з чітким розділенням відповідальності між компонентами.

### Діаграма класів

```mermaid
classDiagram
    %% Діаграма класів основної архітектури програми SSA/ARIMA

    class TimeSeriesDataset {
        +values: np.ndarray
        +name: str
        +units: str
        +source: str
        +from_csv(file_path, column, name, units)
        +from_weather_example(n_points, seed)
    }

    class SSA {
        <<library>>
        +original_series: np.ndarray
        +L: int
        +K: int
        +trajectory_matrix
        +U
        +S
        +V
        +embed()
        +decompose()
        +reconstruct(groups)
        +get_contributions(n)
        +forecast(steps, use_components)
    }

    class SSAAnalyzer {
        +dataset: TimeSeriesDataset
        +window_length: int
        +ssa: SSA
        +components
        +contributions
        +analyze()
        +plot(save_path)
    }

    class ARIMAAnalyzer {
        +dataset: TimeSeriesDataset
        +order: tuple
        +forecast_steps: int
        +results
        +analyze()
        +plot(save_path)
    }

    class ForecastPipeline {
        +dataset: TimeSeriesDataset
        +window_length: int
        +arima_order: tuple
        +forecast_steps: int
        +output_dir: str
        +ssa_analyzer: SSAAnalyzer
        +arima_analyzer: ARIMAAnalyzer
        +run(show_plots)
    }

    class RocketLaunch {
        <<dataclass>>
        +launch_date: str
        +rocket_name: str
        +payload_mass: float
        +orbit: str
        +metric_value: float
    }

    class RocketLaunchDB {
        +db_path: str
        +connect()
        +create_tables()
        +insert_launch(launch: RocketLaunch)
        +ensure_demo_data()
        +load_series(metric) np.ndarray
    }

    %% Зв'язки між класами
    TimeSeriesDataset "1" --> "*" SSAAnalyzer : provides values
    TimeSeriesDataset "1" --> "*" ARIMAAnalyzer : provides values
    SSAAnalyzer "1" --> "1" SSA : uses
    ForecastPipeline "1" --> "1" SSAAnalyzer : композиція
    ForecastPipeline "1" --> "1" ARIMAAnalyzer : композиція
    RocketLaunchDB "1" --> "*" RocketLaunch : зберігає
    RocketLaunchDB ..> TimeSeriesDataset : формує часовий ряд
```

### Діаграма послідовності (основний сценарій)

```mermaid
sequenceDiagram
    %% Основний сценарій роботи програми (користувач -> SSA/ARIMA)

    participant User as Користувач
    participant CLI as main.py / CLI
    participant DS as TimeSeriesDataset
    participant DB as RocketLaunchDB
    participant Pipe as ForecastPipeline
    participant SSAa as SSAAnalyzer
    participant ARIMAa as ARIMAAnalyzer
    participant SSA as SSA (library)

    User->>CLI: запуск програми (python main.py ...)
    CLI->>CLI: parse_args()

    alt Використати БД запусків (--use-db)
        CLI->>DB: RocketLaunchDB(db_path)\nconnect()
        CLI->>DB: create_tables()
        CLI->>DB: ensure_demo_data()
        CLI->>DB: load_series(metric)
        DB-->>CLI: np.ndarray (часовий ряд)
        CLI->>DS: TimeSeriesDataset(values, name="Запуски ракет", source="SQLite БД")
    else CSV / тестові дані
        alt CSV (--data)
            CLI->>DS: TimeSeriesDataset.from_csv(path, column)
        else Тестові дані (погода)
            CLI->>DS: TimeSeriesDataset.from_weather_example(points, seed)
        end
    end

    CLI->>Pipe: ForecastPipeline(dataset, window_length,\n arima_order, forecast_steps, output_dir)

    CLI->>Pipe: run(show_plots)

    activate Pipe
    Pipe->>SSAa: analyze()
    activate SSAa
    SSAa->>SSA: embed() / decompose() / reconstruct()
    SSA-->>SSAa: компоненти, внесок
    SSAa-->>Pipe: SSA результати
    deactivate SSAa

    Pipe->>ARIMAa: analyze()
    activate ARIMAa
    ARIMAa-->>Pipe: ARIMA результати
    deactivate ARIMAa

    Pipe->>Pipe: побудова графіків\nSSA / ARIMA / порівняння
    Pipe-->>CLI: результати та шляхи до зображень
    deactivate Pipe

    CLI-->>User: текстовий звіт + файли графіків
```

### Діаграма модуля роботи з базою даних

```mermaid
classDiagram
    %% Деталізована діаграма класів для модуля роботи з БД запусків ракет

    class RocketLaunch {
        <<dataclass>>
        +launch_date: str
        +rocket_name: str
        +payload_mass: float
        +orbit: str
        +metric_value: float
    }

    class RocketLaunchDB {
        +db_path: str
        +conn: sqlite3.Connection
        +connect()
        +create_tables()
        +insert_launch(launch: RocketLaunch)
        +ensure_demo_data()
        +load_series(metric) np.ndarray
    }

    RocketLaunchDB "1" --> "*" RocketLaunch : вставляє / читає
```

## Структура проекту

```
.
├── main.py                 # Головна програма з CLI інтерфейсом
├── ssa_caterpillar.py      # Бібліотека реалізації методу SSA (Гусениця)
├── rocket_db.py            # Модуль роботи з базою даних запусків ракет
├── requirements.txt        # Залежності Python
├── README.md              # Документація проекту
├── uml_classes.mmd         # Діаграма класів (Mermaid)
├── uml_sequence_main.mmd   # Діаграма послідовності (Mermaid)
├── uml_db_detail.mmd       # Діаграма БД (Mermaid)
└── *.png                   # Згенеровані графіки результатів
```

## Результати

### SSA Аналіз (Метод Гусениця)

Декомпозиція часового ряду на тренд, періодичні компоненти та шум:

![SSA Результати](ssa_results.png)

### ARIMA Прогнозування

Прогноз з 95% довірчим інтервалом та аналіз залишків:

![ARIMA Результати](arima_results.png)

### Порівняння методів

SSA реконструкція vs ARIMA прогноз:

![Порівняння](comparison.png)

## Вихідні файли

Програма генерує три графіки:

- `ssa_results.png` — результати SSA аналізу (тренд, періодика, шум, реконструкція)
- `arima_results.png` — результати ARIMA (прогноз з довірчим інтервалом, залишки)
- `comparison.png` — порівняння методів SSA та ARIMA

## Опис методів

### SSA (Метод Гусениця)

Сингулярний спектральний аналіз розкладає часовий ряд на компоненти:

1. **Вкладення (Embedding)** — побудова траєкторної матриці ковзним вікном
2. **SVD розкладання** — сингулярне розкладання матриці
3. **Групування** — об'єднання компонент за змістом
4. **Реконструкція** — відновлення ряду з обраних компонент

Виділяє: тренд, періодичні компоненти, шум.

### ARIMA

ARIMA(p, d, q) — авторегресійна інтегрована модель ковзного середнього:

- **p** — порядок авторегресії (AR)
- **d** — порядок інтегрування (диференціювання)
- **q** — порядок ковзного середнього (MA)

Використовується для прогнозування часових рядів.

## Формат вхідних даних

CSV файл з одним стовпцем числових значень:

```csv
value
10.5
11.2
12.1
...
```

або з декількома стовпцями (вкажіть потрібний через `--column`):

```csv
date,temperature,humidity
2024-01-01,15.2,65
2024-01-02,14.8,70
...
```

## Середовище розробки

- **Мова:** Python 3.x
- **Бібліотеки:** numpy, pandas, matplotlib, scipy, statsmodels, sqlite3
- **Архітектура:** Модульна з розділенням відповідальності між компонентами
- **Бібліотека SSA:** Власна реалізація методу Caterpillar-SSA (`ssa_caterpillar.py`)

## Додаткова інформація

### Робота з базою даних

Програма підтримує роботу з базою даних SQLite для зберігання та аналізу даних про запуски ракет-носіїв. База даних автоматично створюється при першому запуску з параметром `--use-db` і заповнюється демонстраційними даними, якщо вона порожня.

Доступні метрики для побудови часового ряду:
- `count_per_year` — кількість запусків на рік
- `avg_payload_per_year` — середня маса корисного навантаження на рік
