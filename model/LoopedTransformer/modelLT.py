"""
601Solutions
"""
from __future__ import annotations

# 유의해야할 점: 기존 Transformer 모델과 달리 이 Proj의 Transformer의 sequence는 데이터의 시간축이 아니라 추론 iteration이다.

"""
Mathmatical Modeling
x=[P,v,d,t,E_L,Pe,H*]   (공정조건 4 + 파생 feature 3)
y=[W,R,A]               (R = D/W)

x→fθ​(x)→y^​0​
r0​=PhysicsEvaluator(x,y^​0​)

h1​=Update(h0​,r0​)
^y​1​=fθ​(x,h1​)

r1​=PhysicsEvaluator(x,y^​1​)
Loop...
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoopOutput:
    predictions: list[torch.Tensor]
    feedbacks: list[torch.Tensor]
    confidences: list[torch.Tensor | None] | None = None


class FeatureTokenizer(nn.Module):
    """
    스칼라값 to 토큰

    Input:
        [B, n_features]   (현재 7)

    Output:
        [B, n_features, d_model]
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.scalar_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.scalar_projection(x.unsqueeze(-1))
        return tokens + self.feature_embedding.unsqueeze(0)

def make_encoder(d_model: int, n_heads: int, n_layers: int, dropout: float):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class LoopedTransformer(nn.Module): # L/T Main

    def __init__(
        self,
        n_features: int = 5,
        state_dim: int = 3,
        feedback_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        loop_steps: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feedback_dim = feedback_dim
        self.d_model = d_model
        self.loop_steps = loop_steps

        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.state_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = make_encoder(d_model, n_heads, n_layers, dropout)

        self.state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, state_dim),
        )

        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.state_update = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        physics_evaluator=None,
        loop_steps: int | None = None,
    ) -> LoopOutput:
        bsz = x.size(0)
        steps = loop_steps or self.loop_steps

        process_tokens = self.tokenizer(x)
        state = self.state_token.expand(bsz, -1, -1)

        predictions = []
        feedbacks = []
        confidences = []

        for _ in range(steps):
            tokens = torch.cat([process_tokens, state], dim=1)
            encoded = self.transformer(tokens)

            state_hidden = encoded[:, -1, :]
            prediction = self.state_head(state_hidden)
            predictions.append(prediction)

            if physics_evaluator is None:
                feedback = torch.zeros_like(prediction)
                confidence = None
            else:
                result = physics_evaluator.evaluate(x, prediction)
                feedback = result.residual
                confidence = result.confidence

            feedbacks.append(feedback)
            confidences.append(confidence)

            feedback_hidden = self.feedback_encoder(feedback)
            # 주요 수정 사항.
            PHYSICS_FEEDBACK_SCALE = 0.25
            state = state + PHYSICS_FEEDBACK_SCALE * (self.state_update(
                torch.cat([state_hidden, feedback_hidden], dim=-1)
            )).unsqueeze(1)

        return LoopOutput(predictions=predictions, feedbacks=feedbacks, confidences=confidences)


class StateMLP(nn.Module):
    """baseline."""

    def __init__(self, n_features=4, state_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x):
        return self.net(x)


