import torch.nn as nn


class RotationSslPolicyModel(nn.Module):
    def __init__(self, input_channels: int = 3, embedding_size: int = 64, number_of_actions: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_size),
            nn.ReLU(),
        )
        self.rotation_head = nn.Linear(embedding_size, 4)
        self.action_head = nn.Linear(embedding_size, number_of_actions)

    def encode(self, observation_batch):
        return self.encoder(observation_batch)

    def forward_rotation(self, observation_batch):
        features = self.encode(observation_batch)
        return self.rotation_head(features)

    def forward_action(self, observation_batch):
        features = self.encode(observation_batch)
        return self.action_head(features)

