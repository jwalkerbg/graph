import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# --- 📝 Анотация ---
print("Интерполация на сигнал със смущения от триак и изчисляване на RMS без замърсените части.")

# --- 🔧 Параметри ---
fs = 10000                     # Честота на дискретизация (Hz)
t = np.linspace(0, 0.5, fs//2) # Време (0.5 сек)
signal = np.sin(2 * np.pi * 50 * t)  # Полезен синусоиден сигнал (50 Hz)

# --- ⚡️ Добавяне на триак смущения ---
triac_triggers = [0.1, 0.2, 0.3, 0.4]  # Моменти на отпушване (секунди)
spike_width_ms = 1
spike_amp = 5

noisy_signal = signal.copy()
for trigger in triac_triggers:
    spike_start = int(trigger * fs)
    spike_end = spike_start + int(spike_width_ms * fs / 1000)
    noisy_signal[spike_start:spike_end] += spike_amp * np.random.randn(spike_end - spike_start)

# --- 🧹 Маскиране на смущения ---
def mask_triac_spikes(sig, trigger_times, fs, window_ms=1):
    masked = sig.copy()
    window_samples = int(window_ms * fs / 1000)
    for t_trigger in trigger_times:
        start = int(t_trigger * fs)
        end = start + window_samples
        masked[start:end] = np.nan
    return masked

masked_signal = mask_triac_spikes(noisy_signal, triac_triggers, fs)

# --- 🔁 Интерполация на липсващите данни ---
def interpolate_nan(signal_with_nan, time_array):
    nan_mask = np.isnan(signal_with_nan)
    valid_t = time_array[~nan_mask]
    valid_y = signal_with_nan[~nan_mask]
    interpolator = interpolate.interp1d(valid_t, valid_y, kind='linear', fill_value="extrapolate")
    return interpolator(time_array)

interpolated_signal = interpolate_nan(masked_signal, t)

# --- 📐 RMS изчисление само от чистите части ---
rms_clean = np.sqrt(np.nanmean(masked_signal**2))

# --- 📈 Визуализация ---
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_signal, label="Смущаващ сигнал", alpha=0.5)
plt.plot(t, interpolated_signal, label="Интерполиран сигнал", linewidth=2)
plt.title(f"Интерполация и RMS изчисление (RMS = {rms_clean:.5f})")
plt.xlabel("Време (s)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
