import numpy as np
from scipy.linalg import svd


class SSA:
    """
    Бібліотечний клас для сингулярного спектрального аналізу (SSA),
    також відомого як метод «Гусениця» (Caterpillar).

    Цей файл можна розглядати як окрему Python-бібліотеку, яку
    підключають з інших програм:

        from ssa_caterpillar import SSA
    """

    def __init__(self, time_series, window_length=None):
        self.original_series = np.array(time_series)
        self.N = len(time_series)

        # L – довжина вікна вкладення.
        if window_length is None:
            self.L = self.N // 2
        else:
            self.L = window_length

        # K = N - L + 1 – кількість стовпців траєкторної матриці.
        self.K = self.N - self.L + 1
        self.trajectory_matrix = None
        self.U = None
        self.S = None
        self.V = None
        self.reconstructed_components = None

    def embed(self):
        """
        Етап 1. Вкладення (Embedding).

        Побудова траєкторної матриці A розміру L x K за допомогою
        ковзного вікна довжини L.
        """
        self.trajectory_matrix = np.zeros((self.L, self.K))
        for i in range(self.K):
            self.trajectory_matrix[:, i] = self.original_series[i:i + self.L]
        return self.trajectory_matrix

    def decompose(self):
        """
        Етап 2. Сингулярне розкладання (SVD).

        Обчислюємо сингулярні значення та вектори траєкторної матриці.
        Відповідає пошуку власних значень λ_k матриці S = A · A^T.
        """
        if self.trajectory_matrix is None:
            self.embed()
        self.U, self.S, Vt = svd(self.trajectory_matrix, full_matrices=False)
        self.V = Vt.T
        return self.U, self.S, self.V

    def _diagonal_averaging(self, matrix):
        """
        Допоміжна процедура: діагональне усереднення.

        Перетворює матрицю назад у часовий ряд, усереднюючи елементи
        вздовж побічних діагоналей (аналог формул g_k^B).
        """
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
        """
        Етапи 3–4. Групування та реконструкція компонент.

        :param groups: список списків індексів сингулярних компонент.
                       Кожна група відповідає окремому часовому ряду
                       (тренд, періодика, шум тощо).
        """
        if self.S is None:
            self.decompose()
        if groups is None:
            groups = [[i] for i in range(min(len(self.S), 10))]
        self.reconstructed_components = []
        for group in groups:
            component_matrix = np.zeros((self.L, self.K))
            for idx in group:
                if idx < len(self.S):
                    component_matrix += self.S[idx] * np.outer(
                        self.U[:, idx], self.V[:, idx]
                    )
            reconstructed = self._diagonal_averaging(component_matrix)
            self.reconstructed_components.append(reconstructed)
        return self.reconstructed_components

    def get_contributions(self, n_components=10):
        """
        Обчислення внеску (енергії) перших n сингулярних компонент у %.
        """
        if self.S is None:
            self.decompose()
        total_variance = np.sum(self.S ** 2)
        contributions = (self.S ** 2) / total_variance * 100
        return contributions[:n_components]

    def forecast(self, steps=10, use_components=None):
        """
        Простий приклад прогнозування на основі виділених компонент SSA.

        Для демонстрації використовуємо лінійну екстраполяцію тренду,
        побудовану за останніми значеннями реконструйованого ряду.
        """
        if use_components is None:
            use_components = [0, 1, 2]
        groups = [use_components]
        reconstructed = self.reconstruct(groups)[0]
        forecast_values = list(reconstructed)
        for _ in range(steps):
            trend = np.polyfit(
                range(len(forecast_values[-20:])),
                forecast_values[-20:],
                1
            )
            next_val = trend[0] * len(forecast_values) + trend[1]
            forecast_values.append(next_val)
        return np.array(forecast_values[-steps:])


__all__ = ["SSA"]


