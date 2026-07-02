import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    An LSTM-based classifier for binary time-series prediction.

    This model uses a multilayer LSTM encoder followed by a linear
    classification head. The output is a raw logit (no sigmoid),
    suitable for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        """
        Initialize the LSTM classifier.

        Args:
            input_size (int): Number of input features per time-step.
            hidden_size (int): Hidden dimension of each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability between layers.
            bidirectional (bool): Use bidirectional LSTM.

        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        final_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Shape (batch, seq_len, input_size)

        Returns:
            Tensor: Raw logits, shape (batch, 1)

        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits
