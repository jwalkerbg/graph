import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

# --- Настройки ---
fs = 1000             # Честота на дискретизация (Hz)
t = np.arange(0, 1, 1/fs)  # Времеви вектор за 1 секунда
signal_value = 1.0
internal_noise_std = 0.05
external_noise_freq = 50   # Смущение от мрежа
external_noise_amp = 0.3   # Амплитуда на смущението

# --- Генериране на сигнала ---
pure_signal = np.full_like(t, signal_value)
internal_noise = np.random.normal(0, internal_noise_std, size=t.shape)
external_noise = external_noise_amp * np.sin(2 * np.pi * external_noise_freq * t)

measured_signal = pure_signal + internal_noise + external_noise

# --- Notch филтър около 50 Hz ---
f0 = 50.0  # Централна честота (Hz)
Q = 30.0   # Quality factor (по-високо Q -> по-тясна лента)
b, a = iirnotch(f0, Q, fs)
filtered_signal = filtfilt(b, a, measured_signal)

# --- Изчисления ---
def power(x): return np.mean(x**2)
def rms(x): return np.sqrt(power(x))

print(f"RMS преди филтър:  {rms(measured_signal):.6f}")
print(f"RMS след филтър:   {rms(filtered_signal):.6f}")

# --- Графики ---
plt.figure(figsize=(12, 6))
plt.plot(t, measured_signal, label="Сигнал + вътрешен + външен шум", alpha=0.4)
plt.plot(t, filtered_signal, label="След notch филтър (50 Hz)", linewidth=2)
plt.axhline(signal_value, color='green', linestyle='--', label="Истински сигнал (DC)")
plt.title("Филтриране на външен синусоиден шум (50 Hz)")
plt.xlabel("Време [s]")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
