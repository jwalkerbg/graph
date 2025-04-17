import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

# --- Настройки ---
fs = 1000                # Sampling frequency (Hz)
duration = 1.0           # Продължителност на сигнала (секунди)
t = np.arange(0, duration, 1/fs)
signal_value = 1.0       # Постоянна стойност на полезния сигнал
internal_noise_std = 0.05

# --- Генерация на сигнал ---
pure_signal = np.full_like(t, signal_value)
internal_noise = np.random.normal(0, internal_noise_std, size=t.shape)

# --- Външни нискочестотни смущения (50Hz + хармоници) ---
external_noise = (
    0.3 * np.sin(2 * np.pi * 50 * t) +
    0.2 * np.sin(2 * np.pi * 100 * t) +
    0.1 * np.sin(2 * np.pi * 150 * t)
)

# Общо измерен сигнал
measured_signal = pure_signal + internal_noise + external_noise

# --- Notch филтриране на 50, 100, 150 Hz ---
def apply_notch(sig, freq, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)

filtered_signal = measured_signal
for f in [50, 100, 150]:
    filtered_signal = apply_notch(filtered_signal, f)

# --- Функции за анализ ---
def power(x): return np.mean(x**2)
def rms(x): return np.sqrt(power(x))

print(f"RMS преди филтър:  {rms(measured_signal):.6f}")
print(f"RMS след филтър:   {rms(filtered_signal):.6f}")

# --- Визуализация ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# --- Времева форма ---
axs[0].plot(t, measured_signal, label="Сигнал + смущения", alpha=0.4)
axs[0].plot(t, filtered_signal, label="След notch филтри", linewidth=2)
axs[0].axhline(signal_value, color='green', linestyle='--', label="Истински сигнал (DC)")
axs[0].set_title("Филтриране на 50 Hz и хармоници (времева област)")
axs[0].set_xlabel("Време [s]")
axs[0].set_ylabel("Амплитуда")
axs[0].grid(True)
axs[0].legend()

# --- FFT спектър ---
def plot_fft(ax, sig, fs, label):
    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(sig)
    fft_magnitude = np.abs(fft_vals) / N
    ax.plot(freqs, fft_magnitude, label=label)

plot_fft(axs[1], measured_signal, fs, "Преди филтър")
plot_fft(axs[1], filtered_signal, fs, "След филтър")
axs[1].set_xlim(0, 250)
axs[1].set_title("Амплитуден спектър (FFT)")
axs[1].set_xlabel("Честота [Hz]")
axs[1].set_ylabel("Амплитуда")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
