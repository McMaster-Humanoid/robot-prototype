import numpy as np
import random

class StairGenerator:
    def __init__(self):
        self.base_height = 0.06
        self.max_height = 0.10

        self.base_depth = 0.5
        self.min_depth = 0.25

        self.base_steps = 10
        self.max_steps = 12

    def generate(self, difficulty):
        difficulty = np.clip(difficulty, 0.0, 1.0)

        step_height = self.base_height + difficulty * (self.max_height - self.base_height)
        step_depth = self.base_depth - difficulty * (self.base_depth - self.min_depth)
        num_steps = int(self.base_steps + difficulty * (self.max_steps - self.base_steps))

        stairs = []
        current_x = 1.5
        current_height = 0.0

        for i in range(num_steps):
            height = step_height + random.uniform(-0.005, 0.005)
            depth = step_depth + random.uniform(-0.01, 0.01)
            current_height += height

            stairs.append({
                "x_start": current_x,
                "depth": depth,
                "height": current_height
            })

            current_x += depth

        return stairs