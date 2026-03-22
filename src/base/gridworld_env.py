"""
Simple 2D Gridworld Environment for Phase 2.
Renders as pixel images for vision-based RL.
"""
import numpy as np
import cv2
import random

class GridWorldEnv:
    """
    A simple gridworld environment.
    - Agent (blue square)
    - Target/Goal (green square)
    - Obstacles (red squares)
    
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    
    # Colors (BGR for OpenCV)
    COLOR_BG = (240, 240, 240)       # Light gray
    COLOR_AGENT = (255, 100, 50)     # Blue
    COLOR_TARGET = (50, 200, 50)     # Green
    COLOR_OBSTACLE = (50, 50, 200)   # Red
    COLOR_GRID = (200, 200, 200)     # Grid lines
    
    def __init__(self, grid_size=8, cell_size=8, num_obstacles=3, max_steps=50):
        """
        Args:
            grid_size (int): Number of cells per side (e.g., 8x8 grid).
            cell_size (int): Pixels per cell (e.g., 8 -> 64x64 image).
            num_obstacles (int): Number of obstacle cells.
            max_steps (int): Max steps before episode ends.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.img_size = grid_size * cell_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        
        self.action_space = 4  # Up, Down, Left, Right
        
        self.reset()
    
    def reset(self):
        """Reset the environment to a new random configuration."""
        self.steps = 0
        
        # Place agent, target, and obstacles randomly (no overlaps)
        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        random.shuffle(all_positions)
        
        self.agent_pos = list(all_positions.pop())
        self.target_pos = list(all_positions.pop())
        self.obstacles = [list(all_positions.pop()) for _ in range(self.num_obstacles)]
        
        return self._render()
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): 0=Up, 1=Down, 2=Left, 3=Right
            
        Returns:
            obs (np.array): Next observation (image).
            reward (float): Reward signal.
            done (bool): Whether episode is finished.
            info (dict): Additional info.
        """
        self.steps += 1
        
        # Calculate new position
        new_pos = self.agent_pos.copy()
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        # Check collision with obstacles
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Check termination conditions
        done = False
        reward = -0.01  # Small negative reward per step (encourages efficiency)
        
        if self.agent_pos == self.target_pos:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -0.5
            done = True
            
        obs = self._render()
        
        # Calculate direction to target (for Phase 3 alignment)
        dx = self.target_pos[1] - self.agent_pos[1]
        dy = self.target_pos[0] - self.agent_pos[0]  # Note: row is y
        
        info = {
            'agent_pos': self.agent_pos,
            'target_pos': self.target_pos,
            'direction': (dx, -dy),  # -dy because image y is inverted
            'steps': self.steps
        }
        
        return obs, reward, done, info
    
    def _render(self):
        """Render the current state as an image."""
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
        img[:] = self.COLOR_BG
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            cv2.line(img, (0, i * self.cell_size), (self.img_size, i * self.cell_size), self.COLOR_GRID, 1)
            cv2.line(img, (i * self.cell_size, 0), (i * self.cell_size, self.img_size), self.COLOR_GRID, 1)
        
        # Draw obstacles
        for obs_pos in self.obstacles:
            self._draw_cell(img, obs_pos, self.COLOR_OBSTACLE)
        
        # Draw target
        self._draw_cell(img, self.target_pos, self.COLOR_TARGET)
        
        # Draw agent
        self._draw_cell(img, self.agent_pos, self.COLOR_AGENT)
        
        return img
    
    def _draw_cell(self, img, pos, color):
        """Draw a filled cell at the given grid position."""
        r, c = pos
        x1 = c * self.cell_size + 1
        y1 = r * self.cell_size + 1
        x2 = (c + 1) * self.cell_size - 1
        y2 = (r + 1) * self.cell_size - 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    def get_direction_to_target(self):
        """Get normalized direction vector from agent to target."""
        dx = self.target_pos[1] - self.agent_pos[1]
        dy = -(self.target_pos[0] - self.agent_pos[0])  # Invert y
        
        # Normalize
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            dx /= dist
            dy /= dist
        
        return dx, dy  # This is (cos, sin) of the direction angle


if __name__ == "__main__":
    # Test the environment
    env = GridWorldEnv(grid_size=8, cell_size=8, num_obstacles=5)
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Save initial state
    cv2.imwrite("gridworld_test.png", obs)
    print("Saved gridworld_test.png")
    
    # Take a few random steps
    for i in range(5):
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}, direction={info['direction']}")
        if done:
            break
    
    cv2.imwrite("gridworld_test_final.png", obs)
    print("Saved gridworld_test_final.png")
