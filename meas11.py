import numpy as np
import matplotlib.pyplot as plt

# --- 📝 Анотация ---
print("Симулация на смущения от триак и премахване на засегнатите проби от сигнала.")

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

# --- 🧹 Изолиране на смущения (маскиране) ---
def mask_triac_spikes(sig, trigger_times, fs, window_ms=1):
    masked = sig.copy()
    window_samples = int(window_ms * fs / 1000)
    for t_trigger in trigger_times:
        start = int(t_trigger * fs)
        end = start + window_samples
        masked[start:end] = np.nan  # или интерполация
    return masked

masked_signal = mask_triac_spikes(noisy_signal, triac_triggers, fs)

# --- 📈 Визуализация ---
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_signal, label="Смущаващ сигнал", alpha=0.5)
plt.plot(t, masked_signal, label="След маскиране на смущения", linewidth=2)
for tx in triac_triggers:
    plt.axvline(tx, color='red', linestyle='--', alpha=0.5, label='Триак' if tx == triac_triggers[0] else "")
plt.title("Симулация на триак смущения и премахване")
plt.xlabel("Време (s)")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 