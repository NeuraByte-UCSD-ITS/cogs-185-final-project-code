import random
from typing import List, Tuple

import numpy as np


class ProceduralGridworldEnv:
    """Gridworld environment with numpy-only renderer (no cv2 dependency)."""

    COLOR_BACKGROUND = np.array([240, 240, 240], dtype=np.uint8)
    COLOR_AGENT = np.array([255, 100, 50], dtype=np.uint8)
    COLOR_TARGET = np.array([50, 200, 50], dtype=np.uint8)
    COLOR_OBSTACLE = np.array([50, 50, 200], dtype=np.uint8)

    def __init__(self, grid_size: int = 8, cell_size: int = 8, num_obstacles: int = 1, max_steps: int = 50):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.image_size = grid_size * cell_size
        self.agent_pos: List[int] = [0, 0]
        self.target_pos: List[int] = [0, 0]
        self.obstacles: List[List[int]] = []
        self.steps = 0

    def reset(self):
        self.steps = 0
        all_positions = [(row, column) for row in range(self.grid_size) for column in range(self.grid_size)]
        random.shuffle(all_positions)
        self.agent_pos = list(all_positions.pop())
        self.target_pos = list(all_positions.pop())
        self.obstacles = [list(all_positions.pop()) for _ in range(self.num_obstacles)]
        return self.render()

    def step(self, action: int):
        self.steps += 1
        new_row, new_column = self.agent_pos
        if action == 0:
            new_row = max(0, new_row - 1)
        elif action == 1:
            new_row = min(self.grid_size - 1, new_row + 1)
        elif action == 2:
            new_column = max(0, new_column - 1)
        elif action == 3:
            new_column = min(self.grid_size - 1, new_column + 1)

        if [new_row, new_column] not in self.obstacles:
            self.agent_pos = [new_row, new_column]

        reward = -0.01
        done = False
        if self.agent_pos == self.target_pos:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -0.5
            done = True

        info = {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "obstacles": self.obstacles,
            "steps": self.steps,
        }
        return self.render(), reward, done, info

    def render(self):
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image[:] = self.COLOR_BACKGROUND
        for obstacle_position in self.obstacles:
            self._draw_cell(image, obstacle_position, self.COLOR_OBSTACLE)
        self._draw_cell(image, self.target_pos, self.COLOR_TARGET)
        self._draw_cell(image, self.agent_pos, self.COLOR_AGENT)
        return image

    def _draw_cell(self, image: np.ndarray, position: List[int], color: np.ndarray):
        row, column = position
        row_start = row * self.cell_size
        row_end = row_start + self.cell_size
        column_start = column * self.cell_size
        column_end = column_start + self.cell_size
        image[row_start:row_end, column_start:column_end] = color

