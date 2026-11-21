import pytest
import torch

from power_quant_strategies.models.lstm import LSTMClassifier


@pytest.mark.parametrize("bidirectional", [False, True])
def test_outputs_one_logit_per_sequence(bidirectional):
    model = LSTMClassifier(input_size=4, hidden_size=8, num_layers=2, bidirectional=bidirectional)
    logits = model(torch.randn(5, 12, 4))  # batch, time steps, features
    assert logits.shape == (5, 1)


def test_dropout_only_between_layers():
    assert LSTMClassifier(input_size=3, num_layers=1, dropout=0.5).lstm.dropout == 0.0
    assert LSTMClassifier(input_size=3, num_layers=2, dropout=0.5).lstm.dropout == 0.5


def test_can_learn_a_simple_rule():
    torch.manual_seed(0)
    x = torch.randn(64, 5, 1)
    y = (x[:, -1, 0] > 0).float().unsqueeze(1)  # the sign of the last step
    model = LSTMClassifier(input_size=1, hidden_size=8, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    first_loss = loss_fn(model(x), y).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss / 2
