import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.procedural_gridworld import crop_egocentric_observation
from src.env.multi_goal_demo_env import MultiGoalDemoEnv


@dataclass
class MultiGoalDataConfig:
    grid_size: int
    obstacle_count: int
    number_of_goals: int
    observation_crop_size: int
    episode_horizon_min: int
    episode_horizon_max: int
    cell_size: int = 8


def _compute_oracle_action(agent_position: List[int], target_position: List[int], obstacles: List[List[int]], grid_size: int) -> int:
    """Greedy action toward current target with obstacle and boundary fallback."""
    agent_row, agent_column = agent_position
    target_row, target_column = target_position

    row_delta = target_row - agent_row
    column_delta = target_column - agent_column

    candidate_actions: List[int] = []
    if abs(row_delta) >= abs(column_delta):
        if row_delta < 0:
            candidate_actions.append(0)  # up
        elif row_delta > 0:
            candidate_actions.append(1)  # down
        if column_delta < 0:
            candidate_actions.append(2)  # left
        elif column_delta > 0:
            candidate_actions.append(3)  # right
    else:
        if column_delta < 0:
            candidate_actions.append(2)
        elif column_delta > 0:
            candidate_actions.append(3)
        if row_delta < 0:
            candidate_actions.append(0)
        elif row_delta > 0:
            candidate_actions.append(1)

    for action in [0, 1, 2, 3]:
        if action not in candidate_actions:
            candidate_actions.append(action)

    obstacle_positions = {tuple(position) for position in obstacles}

    def _next_position(action: int) -> Tuple[int, int]:
        next_row, next_column = agent_row, agent_column
        if action == 0:
            next_row = max(0, next_row - 1)
        elif action == 1:
            next_row = min(grid_size - 1, next_row + 1)
        elif action == 2:
            next_column = max(0, next_column - 1)
        elif action == 3:
            next_column = min(grid_size - 1, next_column + 1)
        return next_row, next_column

    for action in candidate_actions:
        if _next_position(action) not in obstacle_positions:
            return action

    return candidate_actions[0]


def generate_sequence_multigoal_imitation_samples(
    number_of_episodes: int,
    data_config: MultiGoalDataConfig,
    seed: int,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    environment = MultiGoalDemoEnv(
        grid_size=data_config.grid_size,
        cell_size=data_config.cell_size,
        num_obstacles=data_config.obstacle_count,
        number_of_goals=data_config.number_of_goals,
        max_steps=data_config.episode_horizon_max,
    )

    sequence_samples: List[np.ndarray] = []
    action_labels: List[int] = []

    for _ in range(number_of_episodes):
        environment.max_steps = random.randint(data_config.episode_horizon_min, data_config.episode_horizon_max)
        full_observation = environment.reset()
        done = False
        steps_taken = 0

        history_buffer = deque(maxlen=sequence_length)

        while not done and steps_taken < environment.max_steps:
            current_cropped_observation = crop_egocentric_observation(
                full_observation,
                environment.agent_pos,
                environment.cell_size,
                data_config.observation_crop_size,
            )
            history_buffer.append(current_cropped_observation)

            if len(history_buffer) < sequence_length:
                pad_frame = history_buffer[0]
                padded_history = [pad_frame] * (sequence_length - len(history_buffer)) + list(history_buffer)
            else:
                padded_history = list(history_buffer)

            oracle_action = _compute_oracle_action(
                environment.agent_pos,
                environment.target_pos,
                environment.obstacles,
                environment.grid_size,
            )

            sequence_samples.append(np.stack(padded_history))
            action_labels.append(oracle_action)

            full_observation, _, done, _ = environment.step(oracle_action)
            steps_taken += 1

    return np.stack(sequence_samples), np.array(action_labels, dtype=np.int64)


class MultiGoalSequenceImitationDataset(Dataset):
    def __init__(self, sequence_array: np.ndarray, action_array: np.ndarray):
        self.sequence_array = sequence_array
        self.action_array = action_array

    def __len__(self):
        return self.action_array.shape[0]

    def __getitem__(self, index: int):
        sequence_tensor = torch.from_numpy(self.sequence_array[index]).float()
        action_tensor = torch.tensor(self.action_array[index], dtype=torch.long)
        return sequence_tensor, action_tensor
