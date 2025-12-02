import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import argparse
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'


class SSA:
    def __init__(self, time_series, window_length=None):
        self.original_series = np.array(time_series)
        self.N = len(time_series)
        
        if window_length is None:
            self.L = self.N // 2
        else:
            self.L = window_length
            
        self.K = self.N - self.L + 1
        self.trajectory_matrix = None
        self.U = None
        self.S = None
        self.V = None
        self.reconstructed_components = None
        
    def embed(self):
        self.trajectory_matrix = np.zeros((self.L, self.K))
        for i in range(self.K):
            self.trajectory_matrix[:, i] = self.original_series[i:i + self.L]
        return self.trajectory_matrix
    
    def decompose(self):
        if self.trajectory_matrix is None:
            self.embed()
        self.U, self.S, Vt = svd(self.trajectory_matrix, full_matrices=False)
        self.V = Vt.T
        return self.U, self.S, self.V
    
    def _diagonal_averaging(self, matrix):
        L, K = matrix.shape
        N = L + K - 1
        result = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                result[i + j] += matrix[i, j]
                counts[i + j] += 1
        return result / counts
    
    def reconstruct(self, groups=None):
        if self.S is None:
            self.decompose()
        if groups is None:
            groups = [[i] for i in range(min(len(self.S), 10))]
        self.reconstructed_components = []
        for group in groups:
            component_matrix = np.zeros((self.L, self.K))
            for idx in group:
                if idx < len(self.S):
                    component_matrix += self.S[idx] * np.outer(self.U[:, idx], self.V[:, idx])
            reconstructed = self._diagonal_averaging(component_matrix)
            self.reconstructed_components.append(reconstructed)
        return self.reconstructed_components
    
    def get_contributions(self, n_components=10):
        if self.S is None:
            self.decompose()
        total_variance = np.sum(self.S ** 2)
        contributions = (self.S ** 2) / total_variance * 100
        return contributions[:n_components]
    
    def forecast(self, steps=10, use_components=None):
        if use_components is None:
            use_components = [0, 1, 2]
        groups = [use_components]
        reconstructed = self.reconstruct(groups)[0]
        forecast_values = list(reconstructed)
        for _ in range(steps):
            trend = np.polyfit(range(len(forecast_values[-20:])), forecast_values[-20:], 1)
            next_val = trend[0] * len(forecast_values) + trend[1]
            forecast_values.append(next_val)
        return np.array(forecast_values[-steps:])


def arima_analysis(time_series, order=(1, 1, 1), forecast_steps=10):
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
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Директорія для збереження графіків (за замовчуванням: поточна)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Не показувати графіки (тільки зберегти)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("СИНГУЛЯРНИЙ СПЕКТРАЛЬНИЙ АНАЛІЗ (SSA) - МЕТОД ГУСЕНИЦЯ (CATERPILLAR)")
    print("ТА ARIMA АНАЛІЗ ЧАСОВИХ РЯДІВ")
    print("=" * 70)
    print()
    
    print("1. ЗАВАНТАЖЕННЯ ДАНИХ")
    print("-" * 40)
    
    if args.data:
        print(f"   Завантаження з файлу: {args.data}")
        time_series = load_csv_data(args.data, args.column)
        print(f"   Завантажено точок: {len(time_series)}")
    else:
        print(f"   Генерація тестових даних...")
        time_series, _, _, _, _ = generate_sample_data(args.points, args.seed)
        print(f"   Згенеровано точок: {len(time_series)}")
        print(f"   Seed: {args.seed}")
    print()
    
    print("2. SSA АНАЛІЗ (МЕТОД ГУСЕНИЦЯ)")
    print("-" * 40)
    
    ssa = SSA(time_series, window_length=args.window)
    
    print(f"   Параметри:")
    print(f"   - Довжина вікна (L): {args.window}")
    print(f"   - Розмір траєкторної матриці: {ssa.L} x {ssa.K}")
    print()
    
    print("   Крок 1: Вкладення (Embedding)")
    trajectory_matrix = ssa.embed()
    print(f"   Траєкторна матриця: {trajectory_matrix.shape}")
    print()
    
    print("   Крок 2: Сингулярне розкладання (SVD)")
    U, S, V = ssa.decompose()
    print(f"   Знайдено сингулярних значень: {len(S)}")
    print(f"   Перші 5: {S[:5].round(2)}")
    print()
    
    contributions = ssa.get_contributions(10)
    print("   Внесок перших 10 компонент:")
    for i, c in enumerate(contributions):
        print(f"   Компонента {i+1}: {c:.2f}%")
    print()
    
    print("   Крок 3-4: Групування та реконструкція")
    groups = [[0], [1, 2], list(range(3, 15))]
    components = ssa.reconstruct(groups)
    print(f"   Виділено груп: {len(groups)}")
    print()
    
    print("3. ARIMA АНАЛІЗ")
    print("-" * 40)
    
    arima_order = (args.arima_p, args.arima_d, args.arima_q)
    
    print(f"   Параметри моделі ARIMA{arima_order}:")
    print(f"   - p = {arima_order[0]} (авторегресія)")
    print(f"   - d = {arima_order[1]} (інтегрування)")
    print(f"   - q = {arima_order[2]} (ковзне середнє)")
    print(f"   - Горизонт прогнозу: {args.forecast} точок")
    print()
    
    arima_results = arima_analysis(time_series, order=arima_order, forecast_steps=args.forecast)
    
    print(f"   Результати:")
    print(f"   - AIC: {arima_results['aic']:.2f}")
    print(f"   - BIC: {arima_results['bic']:.2f}")
    print(f"   - Тест Дікі-Фуллера p-value: {arima_results['adf_pvalue']:.4f}")
    print(f"   - Ряд стаціонарний: {'Так' if arima_results['is_stationary'] else 'Ні'}")
    print()
    
    print("4. ВІЗУАЛІЗАЦІЯ")
    print("-" * 40)
    
    output_dir = args.output_dir.rstrip('/\\')
    
    ssa_path = f"{output_dir}/ssa_results.png"
    arima_path = f"{output_dir}/arima_results.png"
    comparison_path = f"{output_dir}/comparison.png"
    
    if args.no_plots:
        plt.ioff()
    
    print(f"   Збереження SSA графіків: {ssa_path}")
    plot_ssa_results(ssa, time_series, components, "SSA Аналіз (Метод Гусениця)", ssa_path)
    
    print(f"   Збереження ARIMA графіків: {arima_path}")
    plot_arima_results(time_series, arima_results, args.forecast, arima_path)
    
    print(f"   Збереження порівняння: {comparison_path}")
    ssa_reconstructed = components[0] + components[1]
    plot_comparison(time_series, ssa_reconstructed, arima_results['forecast'], args.forecast, comparison_path)
    
    print()
    print("=" * 70)
    print("Аналіз завершено!")
    print("=" * 70)
    
    return ssa, arima_results, components


if __name__ == "__main__":
    ssa, arima_results, components = main()
