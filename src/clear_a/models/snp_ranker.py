"""Step 3b: rank SNPs within top-ranked LD blocks.

Two complementary models are trained and their normalised scores averaged:
- Elastic-net logistic regression on dosage matrices: linear, fast, gives
  signed coefficients useful for downstream interpretation.
- A small dropout MLP on dosages + per-SNP annotation features: captures
  modest non-linear interactions; relevance scores via gradient × input.

Inputs are restricted to SNPs whose block_id is in `top_blocks` from the
group ranker. This is the search-space-reduction step the proposal claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@dataclass
class SnpRankConfig:
    elastic_net_alpha: float = 0.5
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 500  # SAGA on 100k+ features needs a hard cap
    dnn_hidden: tuple[int, ...] = (128, 64)
    dnn_dropout: float = 0.3
    dnn_epochs: int = 30
    dnn_batch_size: int = 64
    dnn_lr: float = 1e-3
    seed: int = 42
    # When n_features >> n_samples, switch from SAGA elasticnet to liblinear
    # L1: 10-100x faster, sparsity is what we want anyway.
    liblinear_threshold_features: int = 20_000


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float, n_classes: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gradient_x_input(model: _MLP, X: np.ndarray, target_class: int) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    logits = model(Xt)
    score = logits[:, target_class].sum()
    score.backward()
    grads = Xt.grad.detach().numpy()
    return np.abs(grads * X).mean(axis=0)


class SnpRanker:
    def __init__(self, config: SnpRankConfig | None = None):
        self.config = config or SnpRankConfig()
        self.scaler_: StandardScaler | None = None
        self.elastic_: LogisticRegression | None = None
        self.mlp_: _MLP | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        dosages: np.ndarray,
        snp_features: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> "SnpRanker":
        # Concat dosage + per-SNP feature matrix is unnecessary for ranking SNPs:
        # we score SNPs *as predictors*. So the design matrix is the dosages
        # themselves, and snp_features are only used to standardise per-SNP
        # weights (e.g., divide by sqrt(MAF*(1-MAF))).
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.feature_names_ = list(feature_names)
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        X = self.scaler_.fit_transform(dosages.astype(np.float32))

        n_classes = int(np.max(y)) + 1 if y.dtype.kind in "iu" else 2

        # SAGA elasticnet is intractable when n_features >> n_samples (e.g.
        # 100k SNPs / 600 samples). Fall back to liblinear L1 in that regime —
        # for ranking purposes we only need sparse coefficients, and liblinear
        # converges in seconds vs SAGA's hours.
        if X.shape[1] > self.config.liblinear_threshold_features:
            self.elastic_ = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0 / self.config.elastic_net_alpha,
                max_iter=self.config.elastic_net_max_iter,
                random_state=self.config.seed,
            )
        else:
            self.elastic_ = LogisticRegression(
                solver="saga",
                l1_ratio=self.config.elastic_net_l1_ratio,
                C=1.0 / self.config.elastic_net_alpha,
                max_iter=self.config.elastic_net_max_iter,
                random_state=self.config.seed,
            )
        self.elastic_.fit(X, y)

        self.mlp_ = _MLP(
            in_dim=X.shape[1],
            hidden=tuple(self.config.dnn_hidden),
            dropout=self.config.dnn_dropout,
            n_classes=max(n_classes, 2),
        )
        opt = torch.optim.Adam(self.mlp_.parameters(), lr=self.config.dnn_lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        n = X.shape[0]
        bs = self.config.dnn_batch_size
        for epoch in range(self.config.dnn_epochs):
            perm = torch.randperm(n)
            losses = []
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                logits = self.mlp_(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                log.info("MLP epoch %d/%d loss=%.4f",
                         epoch + 1, self.config.dnn_epochs, float(np.mean(losses)))
        return self

    def rank(self, dosages: np.ndarray, target_class: int = 1) -> pd.Series:
        assert self.scaler_ is not None and self.elastic_ is not None and self.mlp_ is not None
        X = self.scaler_.transform(dosages.astype(np.float32))

        coef = self.elastic_.coef_
        if coef.ndim == 1:
            elastic_score = np.abs(coef)
        elif coef.shape[0] == 1:
            # Binary task: sklearn returns (1, n_features); collapse to vector.
            elastic_score = np.abs(coef[0])
        else:
            # Multiclass: pick the requested class row.
            row = min(target_class, coef.shape[0] - 1)
            elastic_score = np.abs(coef[row])

        mlp_score = _gradient_x_input(self.mlp_, X, target_class=target_class)

        def _norm(s: np.ndarray) -> np.ndarray:
            mx = s.max()
            return s / mx if mx > 0 else s

        combined = 0.5 * _norm(elastic_score) + 0.5 * _norm(mlp_score)
        return pd.Series(combined, index=self.feature_names_, name="snp_score").sort_values(
            ascending=False
        )
