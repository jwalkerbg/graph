import time
import random
import matplotlib.pyplot as plt

class MagneticSensorSmoothing:
    def __init__(self, alpha=0.1):
        """
        Initialize the smoothing algorithm.
        :param alpha: Smoothing factor (0 < alpha <= 1).
                      Smaller values result in slower response.
        """
        self.alpha = alpha
        self.real_setting = None  # This will store the smoothed value

    def update(self, input_value):
        """
        Update the smoothed value based on the input.
        :param input_value: The new input value from the sensor (0 to 270 degrees).
        :return: Smoothed real setting.
        """
        if self.real_setting is None:
            # First input initializes the real setting
            self.real_setting = input_value
        else:
            # Apply exponential smoothing
            self.real_setting += self.alpha * (input_value - self.real_setting)
        return self.real_setting

if __name__ == "__main__":
    # Create an instance of the smoothing algorithm
    smoother = MagneticSensorSmoothing(alpha=0.05)

    # Initialize a fixed input value
    input_value = random.uniform(0, 270)

    # Prepare for plotting
    iterations = 1000
    x_vals = []  # Iteration numbers
    input_vals = []  # Input values
    smoothed_vals = []  # Smoothed values

    plt.ion()  # Interactive mode for real-time updating
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Input Value", color="blue")
    line2, = ax.plot([], [], label="Smoothed Value", color="orange")
    ax.set_xlim(0, iterations)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value (degrees)")
    ax.legend()

    for i in range(iterations):
        if i % 70 == 0:  # Change input value every 100th iteration
            input_value = random.uniform(0, 270)

        smoothed_value = smoother.update(input_value)

        # Update data for plotting
        x_vals.append(i)
        input_vals.append(input_value)
        smoothed_vals.append(smoothed_value)

        # Update the plot
        line1.set_data(x_vals, input_vals)
        line2.set_data(x_vals, smoothed_vals)
        ax.set_xlim(0, max(10, i))  # Dynamically adjust x-axis
        plt.pause(0.01)  # Pause for a brief moment to update the plot

    plt.ioff()  # Turn off interactive mode
    plt.show()
