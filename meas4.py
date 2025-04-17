import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

# --- Настройки ---
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)
signal_value = 1.0
internal_noise_std = 0.05

# --- Вътрешен сигнал ---
pure_signal = np.full_like(t, signal_value)
internal_noise = np.random.normal(0, internal_noise_std, size=t.shape)

# --- Външен смущаващ шум: 100 Hz + хармоници (200 Hz, 300 Hz) ---
external_noise = (
    0.3 * np.sin(2 * np.pi * 100 * t) +
    0.2 * np.sin(2 * np.pi * 200 * t) +
    0.1 * np.sin(2 * np.pi * 300 * t)
)

measured_signal = pure_signal + internal_noise + external_noise

# --- Прилагане на няколко notch филтъра ---
def apply_notch(sig, freq, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)

filtered_signal = measured_signal
for f in [100, 200, 300]:
    filtered_signal = apply_notch(filtered_signal, f)

# --- Метрики ---
def power(x): return np.mean(x**2)
def rms(x): return np.sqrt(power(x))

print(f"RMS преди филтър:  {rms(measured_signal):.6f}")
print(f"RMS след филтър:   {rms(filtered_signal):.6f}")

# --- Визуализация ---
plt.figure(figsize=(12, 6))
plt.plot(t, measured_signal, label="Сигнал + шум (100Hz, 200Hz, 300Hz)", alpha=0.4)
plt.plot(t, filtered_signal, label="След notch филтри", linewidth=2)
plt.axhline(signal_value, color='green', linestyle='--', label="Истински сигнал (DC)")
plt.title("Филтриране на хармонично външно смущение")
plt.xlabel("Време [s]")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
