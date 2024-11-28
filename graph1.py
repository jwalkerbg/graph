import sys
import matplotlib
matplotlib.use('QtAgg')  # Use the QtAgg backend for PyQt6
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton

# Parameters
N = 50  # Number of records to display
update_interval = 0.1  # Time interval between updates (seconds)

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

def generate_data():
    """Simulate real-time data generation."""
    return np.sin(time.time()) + np.random.normal(scale=0.1)

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

# Global flag to control the data generation
running = True

def start_generation():
    """Start data generation."""
    global running
    running = True
    print("Data generation started.")

def stop_generation():
    """Stop data generation."""
    global running
    running = False
    print("Data generation stopped.")

start_button.clicked.connect(start_generation)
stop_button.clicked.connect(stop_generation)

# Update the plot in real time
def update_plot():
    global running
    while True:
        if running:
            # Generate and append new data
            new_value = generate_data()
            data.append(new_value)

            # Update the line plot
            line.set_ydata(data)
            ax.set_ylim(min(data) - 0.1, max(data) + 0.1)  # Dynamic Y-axis scaling if needed
            plt.draw()
            plt.pause(update_interval)

            # Handle PyQt6 events
            app.processEvents()

# Start the plot update in a separate thread
import threading
plot_thread = threading.Thread(target=update_plot, daemon=True)
plot_thread.start()

# Show the window
window.show()

# Run the PyQt6 application event loop
sys.exit(app.exec())
