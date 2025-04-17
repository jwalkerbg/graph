import numpy as np
import matplotlib.pyplot as plt

# === Анотация ===
print("Пример: Използване на адаптивен LMS филтър за премахване на 50 Hz смущения от измерен сигнал.\n"
      "Изход: графики на измерения сигнал, оценения шум и филтрирания сигнал.")

# --- Настройки ---
fs = 1000                # честота на дискретизация (Hz)
duration = 1.0           # време на сигнала в секунди
t = np.arange(0, duration, 1/fs)

signal_value = 1.0
internal_noise_std = 0.05

# --- Генериране на полезен сигнал ---
pure_signal = np.full_like(t, signal_value)
internal_noise = np.random.normal(0, internal_noise_std, size=t.shape)

# --- Външен шум (50Hz + хармоници) ---
external_noise = (
    0.3 * np.sin(2 * np.pi * 50 * t) +
    0.2 * np.sin(2 * np.pi * 100 * t) +
    0.1 * np.sin(2 * np.pi * 150 * t)
)

# Измерен сигнал = сигнал + вътрешен + външен шум
measured_signal = pure_signal + internal_noise + external_noise

# Референтен шумов вход (знаем формата на външния шум)
reference_noise = external_noise

# --- LMS Филтър ---
def lms_filter(x, d, mu=0.01, filter_order=16):
    """
    x: референтен вход (шум)
    d: сигнал със смущения
    mu: скорост на учене
    """
    N = len(x)
    y = np.zeros(N)
    e = np.zeros(N)
    w = np.zeros(filter_order)

    for n in range(filter_order, N):
        x_vec = x[n-filter_order:n][::-1]
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w += 2 * mu * e[n] * x_vec  # LMS актуализация
    return e, y

# --- Изпълнение ---
output_signal, estimated_noise = lms_filter(reference_noise, measured_signal)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

axs[0].plot(t, measured_signal, label="Сигнал + шум", alpha=0.5)
axs[0].plot(t, pure_signal, '--', label="Истински сигнал")
axs[0].set_title("Измерен сигнал (с шум)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, estimated_noise, label="LMS оценен шум", color='orange')
axs[1].set_title("Оценен шум от LMS")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, output_signal, label="LMS изход (очистен)", color='green')
axs[2].plot(t, pure_signal, '--', label="Истински сигнал", color='black')
axs[2].set_title("Сигнал след LMS филтриране")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
