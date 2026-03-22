import torch
import torch.nn as nn


class PredictiveSslPolicyModel(nn.Module):
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
        self.predictor_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )
        self.action_head = nn.Linear(embedding_size, number_of_actions)

    def encode(self, observation_batch: torch.Tensor):
        return self.encoder(observation_batch)

    def forward_predict_next_embedding(self, observation_batch: torch.Tensor):
        current_embedding = self.encode(observation_batch)
        predicted_next_embedding = self.predictor_head(current_embedding)
        return predicted_next_embedding

    def forward_action(self, observation_batch: torch.Tensor):
        embedding = self.encode(observation_batch)
        return self.action_head(embedding)
