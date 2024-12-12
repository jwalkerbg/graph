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
    import time
    import random

    # Create an instance of the smoothing algorithm
    smoother = MagneticSensorSmoothing(alpha=0.05)

    # Initialize a fixed input value
    input_value = random.uniform(0, 270)

    # Simulate sensor input with change every 100 iterations
    for i in range(1000):  # Increased loop count for demonstration
        if i % 100 == 0:  # Change input value every 100th iteration
            input_value = random.uniform(0, 270)

        smoothed_value = smoother.update(input_value)
        print(f"Iteration: {i}, Input: {input_value:.2f}, Smoothed: {smoothed_value:.2f}")
        time.sleep(0.01)  # Simulate a delay between readings
