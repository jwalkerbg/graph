import sys
import matplotlib
matplotlib.use('QtAgg')  # Use the QtAgg backend for PyQt6
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import QTimer

# Parameters
N = 200  # Number of records to display
update_interval = 100  # Time interval between updates (milliseconds)

# Initialize data storage
data = deque([0] * N, maxlen=N)  # A fixed-length deque to store the last N records
x = np.arange(N)  # X-axis indices

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, data)
ax.set_ylim(-1, 1)  # Adjust based on the range of generated data
ax.set_title("Real-Time Data Plot")
ax.set_xlabel("Time (Relative Index)")
ax.set_ylabel("Value")

iteration = 0.0
step = 0.09
scale = 0.01
def generate_data():
    global iteration,scale, step
    """Simulate real-time data generation."""

    ret = np.sin(iteration) + np.random.normal(scale=scale)
    iteration += step
    return ret

# PyQt6 application setup
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Real-Time Data Plot")

# Layout and buttons
layout = QVBoxLayout()

start_button = QPushButton("Start Generation")
stop_button = QPushButton("Stop Generation")

layout.addWidget(start_button)
layout.addWidget(stop_button)
window.setLayout(layout)

# Timer for updating the plot
timer = QTimer()
timer.setInterval(update_interval)

def update_plot():
    """Update the plot with new data."""
    new_value = generate_data()
    data.append(new_value)

    # Update the line plot
    line.set_ydata(data)
    ax.set_ylim(min(data) - 0.1, max(data) + 0.1)  # Dynamic Y-axis scaling if needed
    plt.draw()

# Button actions
def start_generation():
    """Start data generation."""
    timer.start()  # Start the timer for updates
    print("Data generation started.")

def stop_generation():
    """Stop data generation."""
    timer.stop()  # Stop the timer for updates
    print("Data generation stopped.")

# Connect buttons to their actions
start_button.clicked.connect(start_generation)
stop_button.clicked.connect(stop_generation)

# Connect the timer to the update_plot function
timer.timeout.connect(update_plot)

# Show the window
window.show()

# Run the PyQt6 application event loop
sys.exit(app.exec())
