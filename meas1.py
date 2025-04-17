import numpy as np
import matplotlib.pyplot as plt

# --- Генериране на проби ---
def generate_noise_samples(n, noise_std=0.05):
    return np.random.normal(loc=0.0, scale=noise_std, size=n)

def generate_signal_samples(n, signal_value=1.0, noise_std=0.05):
    noise = generate_noise_samples(n, noise_std)
    signal = np.full(n, signal_value)
    return signal + noise

# --- Средна мощност (RMS^2) ---
def mean_power(samples):
    return np.mean(samples ** 2)

# --- Филтър с движещо се средно ---
def moving_average_filter(samples, window_size=5):
    return np.convolve(samples, np.ones(window_size)/window_size, mode='valid')

# --- Симулация ---
N = 1000
SIGNAL_VALUE = 1.0
NOISE_STD = 0.1
WINDOW = 10

# 1. Измерваме само шума
noise_samples = generate_noise_samples(N, noise_std=NOISE_STD)
noise_power = mean_power(noise_samples)

# 2. Измерваме сигнал + шум
raw_signal_samples = generate_signal_samples(N, signal_value=SIGNAL_VALUE, noise_std=NOISE_STD)
raw_measured_power = mean_power(raw_signal_samples)

# 3. Прилагаме филтър
filtered_signal = moving_average_filter(raw_signal_samples, window_size=WINDOW)
filtered_power = mean_power(filtered_signal)

# 4. Изчисляваме сигнална мощност
raw_signal_power = raw_measured_power - noise_power
filtered_signal_power = filtered_power - noise_power  # ще приемем, че шумът е все още същия

# 5. RMS
raw_signal_rms = np.sqrt(raw_signal_power)
filtered_signal_rms = np.sqrt(filtered_signal_power)

# --- Резултати ---
print(f"Без филтър:")
print(f"  Измерена обща мощност:     {raw_measured_power:.6f}")
print(f"  Сигнална мощност (оценка): {raw_signal_power:.6f}")
print(f"  RMS на сигнала:             {raw_signal_rms:.6f}")

print(f"\nС филтър (moving avg, {WINDOW} проби):")
print(f"  Филтрирана обща мощност:    {filtered_power:.6f}")
print(f"  Сигнална мощност (оценка):  {filtered_signal_power:.6f}")
print(f"  RMS на сигнала:             {filtered_signal_rms:.6f}")

# --- Графика ---
plt.figure(figsize=(10, 4))
plt.plot(raw_signal_samples, label='Сигнал + шум (нефилтриран)', alpha=0.5)
plt.plot(np.arange(WINDOW-1, N), filtered_signal, label='Филтриран сигнал', linewidth=2)
plt.axhline(SIGNAL_VALUE, color='green', linestyle='--', label='Истинска стойност на сигнала')
plt.title("Филтриране с движещо се средно")
plt.xlabel("Измерване")
plt.ylabel("Стойност")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
