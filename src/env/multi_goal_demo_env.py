import random
from typing import List, Tuple

import numpy as np


class MultiGoalDemoEnv:
    COLOR_BACKGROUND = np.array([240, 240, 240], dtype=np.uint8)
    COLOR_AGENT = np.array([255, 100, 50], dtype=np.uint8)
    COLOR_TARGET = np.array([50, 200, 50], dtype=np.uint8)
    COLOR_OBSTACLE = np.array([50, 50, 200], dtype=np.uint8)

    def __init__(
        self,
        grid_size: int = 10,
        cell_size: int = 8,
        num_obstacles: int = 3,
        number_of_goals: int = 3,
        max_steps: int = 120,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_obstacles = num_obstacles
        self.number_of_goals = number_of_goals
        self.max_steps = max_steps
        self.image_size = grid_size * cell_size
        self.agent_pos: List[int] = [0, 0]
        self.goal_positions: List[List[int]] = []
        self.current_goal_index = 0
        self.obstacles: List[List[int]] = []
        self.steps = 0

    @property
    def target_pos(self) -> List[int]:
        return self.goal_positions[self.current_goal_index]

    def reset(self):
        self.steps = 0
        self.current_goal_index = 0
        all_positions = [(row, column) for row in range(self.grid_size) for column in range(self.grid_size)]
        random.shuffle(all_positions)
        self.agent_pos = list(all_positions.pop())
        self.obstacles = [list(all_positions.pop()) for _ in range(self.num_obstacles)]
        self.goal_positions = [list(all_positions.pop()) for _ in range(self.number_of_goals)]
        return self.render()

    def _move_agent(self, action: int):
        next_row, next_column = self.agent_pos
        if action == 0:
            next_row = max(0, next_row - 1)
        elif action == 1:
            next_row = min(self.grid_size - 1, next_row + 1)
        elif action == 2:
            next_column = max(0, next_column - 1)
        elif action == 3:
            next_column = min(self.grid_size - 1, next_column + 1)

        if [next_row, next_column] not in self.obstacles:
            self.agent_pos = [next_row, next_column]

    def step(self, action: int):
        self.steps += 1
        self._move_agent(action)
        reward = -0.01
        done = False
        reached_goal = False

        if self.agent_pos == self.target_pos:
            reached_goal = True
            reward = 1.0
            if self.current_goal_index < self.number_of_goals - 1:
                self.current_goal_index += 1
            else:
                done = True
        elif self.steps >= self.max_steps:
            reward = -0.5
            done = True

        info = {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "goal_positions": self.goal_positions,
            "current_goal_index": self.current_goal_index,
            "obstacles": self.obstacles,
            "steps": self.steps,
            "reached_goal": reached_goal,
        }
        return self.render(), reward, done, info

    def render(self):
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image[:] = self.COLOR_BACKGROUND
        for obstacle_position in self.obstacles:
            self._draw_cell(image, obstacle_position, self.COLOR_OBSTACLE)
        for goal_index, goal_position in enumerate(self.goal_positions):
            color = self.COLOR_TARGET if goal_index == self.current_goal_index else np.array([120, 220, 120], dtype=np.uint8)
            self._draw_cell(image, goal_position, color)
        self._draw_cell(image, self.agent_pos, self.COLOR_AGENT)
        return image

    def _draw_cell(self, image: np.ndarray, position: List[int], color: np.ndarray):
        row, column = position
        row_start = row * self.cell_size
        row_end = row_start + self.cell_size
        column_start = column * self.cell_size
        column_end = column_start + self.cell_size
        image[row_start:row_end, column_start:column_end] = color

