import torch
import torch.nn as nn


class RecurrentPolicyNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        number_of_actions: int = 4,
        embedding_size: int = 64,
        lstm_hidden_size: int = 64,
        conv_depth: int = 3,
    ):
        super().__init__()
        conv_depth = max(3, int(conv_depth))
        encoder_layers = []
        in_channels = input_channels
        out_channels_by_layer = [32] + [64] * (conv_depth - 1)
        for out_channels in out_channels_by_layer:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU())
            in_channels = out_channels
        encoder_layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, embedding_size),
                nn.ReLU(),
            ]
        )
        self.frame_encoder = nn.Sequential(*encoder_layers)
        self.sequence_model = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Linear(lstm_hidden_size, number_of_actions)

    def forward(self, sequence_batch: torch.Tensor):
        # sequence_batch: [B, T, C, H, W]
        batch_size, sequence_length, channels, height, width = sequence_batch.shape
        flattened_frames = sequence_batch.view(batch_size * sequence_length, channels, height, width)
        encoded_frames = self.frame_encoder(flattened_frames)
        encoded_sequences = encoded_frames.view(batch_size, sequence_length, -1)
        lstm_outputs, _ = self.sequence_model(encoded_sequences)
        final_step_features = lstm_outputs[:, -1, :]
        return self.action_head(final_step_features)
