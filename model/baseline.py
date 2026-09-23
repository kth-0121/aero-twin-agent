"""Physics-agnostic baseline, for measuring what the physics-residual actually buys us."""
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def demo():
    torch.manual_seed(0)
    model = BaselineMLP(input_dim=3, output_dim=1)
    x = torch.randn(5, 3)
    pred = model(x)
    assert pred.shape == (5, 1)
    print(f"demo ok: pred shape={tuple(pred.shape)}")


if __name__ == "__main__":
    demo()
