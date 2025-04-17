import numpy as np
import matplotlib.pyplot as plt

# --- Генериране на проби ---
def generate_noise_samples(n, noise_std=0.05):
    return np.random.normal(loc=0.0, scale=noise_std, size=n)

def generate_signal_samples(n, signal_value=1.0, noise_std=0.05):
    noise = generate_noise_samples(n, noise_std)
    signal = np.full(n, signal_value)
    return signal + noise

# --- RMS мощност ---
def mean_power(samples):
    return np.mean(samples ** 2)

# --- Филтри с движещо се средно и експоненциален ---
def moving_average_filter(samples, window_size=5):
    return np.convolve(samples, np.ones(window_size)/window_size, mode='valid')

def exponential_filter(samples, alpha=0.1):
    filtered = np.zeros_like(samples)
    filtered[0] = samples[0]
    for i in range(1, len(samples)):
        filtered[i] = alpha * samples[i] + (1 - alpha) * filtered[i - 1]
    return filtered

# --- Настройки ---
N = 1000
SIGNAL_VALUE = 1.0
NOISE_STD = 0.1
WINDOW = 10
ALPHA = 0.1

# --- Шум ---
noise_samples = generate_noise_samples(N, noise_std=NOISE_STD)
noise_power = mean_power(noise_samples)

# --- Сигнал + шум ---
raw_signal_samples = generate_signal_samples(N, signal_value=SIGNAL_VALUE, noise_std=NOISE_STD)
raw_measured_power = mean_power(raw_signal_samples)
raw_signal_power = raw_measured_power - noise_power
raw_signal_rms = np.sqrt(raw_signal_power)

# --- Moving Average ---
ma_signal = moving_average_filter(raw_signal_samples, window_size=WINDOW)
ma_power = mean_power(ma_signal)
ma_signal_power = ma_power - noise_power
ma_signal_rms = np.sqrt(ma_signal_power)

# --- Експоненциален филтър ---
exp_signal = exponential_filter(raw_signal_samples, alpha=ALPHA)
exp_power = mean_power(exp_signal)
exp_signal_power = exp_power - noise_power
exp_signal_rms = np.sqrt(exp_signal_power)

# --- Резултати ---
print(f"Без филтър:")
print(f"  Измерена обща мощност:     {raw_measured_power:.6f}")
print(f"  Сигнална мощност (оценка): {raw_signal_power:.6f}")
print(f"  RMS на сигнала:             {raw_signal_rms:.6f}")

print(f"\nС Moving Average (прозорец {WINDOW}):")
print(f"  Филтрирана обща мощност:    {ma_power:.6f}")
print(f"  Сигнална мощност (оценка):  {ma_signal_power:.6f}")
print(f"  RMS на сигнала:             {ma_signal_rms:.6f}")

print(f"\nС Експоненциален филтър (alpha={ALPHA}):")
print(f"  Филтрирана обща мощност:    {exp_power:.6f}")
print(f"  Сигнална мощност (оценка):  {exp_signal_power:.6f}")
print(f"  RMS на сигнала:             {exp_signal_rms:.6f}")

# --- Графики ---
plt.figure(figsize=(12, 6))
plt.plot(raw_signal_samples, label='Сигнал + шум (нефилтриран)', alpha=0.4)
plt.plot(np.arange(WINDOW-1, N), ma_signal, label='Moving Average', linewidth=2)
plt.plot(exp_signal, label='Експоненциален филтър', linewidth=2)
plt.axhline(SIGNAL_VALUE, color='green', linestyle='--', label='Истинска стойност')
plt.title("Сравнение на филтри")
plt.xlabel("Проби")
plt.ylabel("Стойност")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
