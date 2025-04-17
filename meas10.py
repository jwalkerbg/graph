import numpy as np
import matplotlib.pyplot as plt

# === Анотация ===
print("Пример: Филтрация в честотната област чрез FFT\n"
      "Цел: Премахване на 50 Hz и хармоници от сигнал с постоянна стойност.\n")

# --- Параметри ---
fs = 1000            # честота на дискретизация
duration = 1.0       # времетраене
t = np.arange(0, duration, 1/fs)

# --- Сигнали ---
pure_signal = np.ones_like(t) * 1.0
internal_noise = np.random.normal(0, 0.05, size=t.shape)
external_noise = (
    0.3 * np.sin(2 * np.pi * 50 * t) +
    0.2 * np.sin(2 * np.pi * 100 * t)
)

measured_signal = pure_signal + internal_noise + external_noise

# --- Преобразуване с FFT ---
N = len(measured_signal)
freqs = np.fft.rfftfreq(N, d=1/fs)
fft_vals = np.fft.rfft(measured_signal)

# --- Филтриране: зануляване около 50 Hz и 100 Hz ---
def notch_filter(freqs, fft_vals, notch_freqs, width=1):
    fft_filtered = fft_vals.copy()
    for f in notch_freqs:
        indices = np.where(np.abs(freqs - f) < width)[0]
        fft_filtered[indices] = 0
    return fft_filtered

fft_filtered = notch_filter(freqs, fft_vals, notch_freqs=[50, 100], width=1.5)

# --- Възстановяване във времева област ---
filtered_signal = np.fft.irfft(fft_filtered, n=N)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

axs[0].plot(t, measured_signal, label="Оригинален сигнал")
axs[0].set_title("Сигнал преди филтрация")
axs[0].grid()
axs[0].legend()

axs[1].plot(freqs, np.abs(fft_vals), label="Спектър преди")
axs[1].plot(freqs, np.abs(fft_filtered), label="След филтрация")
axs[1].set_title("Амплитуден спектър")
axs[1].set_xlim(0, 200)
axs[1].grid()
axs[1].legend()

axs[2].plot(t, filtered_signal, label="Филтриран сигнал", color='green')
axs[2].set_title("Сигнал след филтрация в честотната област")
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()
