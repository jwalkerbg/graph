import numpy as np
import matplotlib.pyplot as plt

# === Анотация ===
print("Пример: LMS адаптивен филтър с множество синусоидални входове (50 Hz и хармоници)\n"
      "Цел: Премахване на нискочестотни смущения от измерен сигнал с постоянна стойност.\n")

# --- Параметри ---
fs = 1000          # Честота на дискретизация (Hz)
duration = 1.0     # Продължителност на сигнала (сек)
t = np.arange(0, duration, 1/fs)

signal_value = 1.0
internal_noise_std = 0.05

# --- Полезен сигнал и шум ---
pure_signal = np.full_like(t, signal_value)
internal_noise = np.random.normal(0, internal_noise_std, size=t.shape)

# --- Външен шум (50Hz + хармоници) ---
external_noise = (
    0.3 * np.sin(2 * np.pi * 50 * t) +
    0.2 * np.sin(2 * np.pi * 100 * t) +
    0.1 * np.sin(2 * np.pi * 150 * t)
)

# --- Измерен сигнал ---
measured_signal = pure_signal + internal_noise + external_noise

# --- Референтен вход: синусоиди с известни честоти ---
ref_freqs = [50, 100, 150]
ref_signals = [np.sin(2 * np.pi * f * t) for f in ref_freqs]

# Обединяваме всички синусоиди в една матрица: (N, M)
X = np.stack(ref_signals, axis=1)  # shape: (samples, num_features)

# --- LMS филтър с векторен вход ---
def vector_lms_filter(X, d, mu=0.01):
    """
    X: (N, M) - N семпъла, M референтни канала (синусоиди)
    d: (N,) измерен сигнал със смущения
    mu: скорост на учене
    """
    N, M = X.shape
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(N):
        x_n = X[n]
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        w += 2 * mu * e[n] * x_n
    return e, y

# --- Прилагане на филтъра ---
filtered_signal, estimated_noise = vector_lms_filter(X, measured_signal, mu=0.01)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

axs[0].plot(t, measured_signal, label="Сигнал с шум", alpha=0.6)
axs[0].plot(t, pure_signal, '--', label="Истински сигнал")
axs[0].set_title("Измерен сигнал (вкл. външен и вътрешен шум)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, estimated_noise, label="Оценен шум от LMS", color='orange')
axs[1].set_title("Оценен външен шум (LMS)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, filtered_signal, label="Филтриран сигнал", color='green')
axs[2].plot(t, pure_signal, '--', label="Истински сигнал", color='black')
axs[2].set_title("Сигнал след LMS филтриране")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
