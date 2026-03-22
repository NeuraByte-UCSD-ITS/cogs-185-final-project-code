import torch.nn as nn


class FeedforwardPolicyNetwork(nn.Module):
    def __init__(self, input_channels: int = 3, number_of_actions: int = 4, embedding_size: int = 64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_size),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(embedding_size, number_of_actions)

    def forward(self, observation_batch):
        extracted_features = self.feature_extractor(observation_batch)
        latent_embedding = self.embedding_layer(extracted_features)
        return self.action_head(latent_embedding)

