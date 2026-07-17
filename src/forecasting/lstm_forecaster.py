"""LSTM demand forecaster (SPEC.md §4) — supporting component only.

Predicts next-hour demand for all 5 zones from a 24-hour window of per-zone
demand plus cyclical time encodings. Its role in the report is to show the
demand process is learnable structure (not noise) and to motivate
forecast-aware shedding as future work; it is NOT wired into the RL agents
(trimmed scope).

Train: python -m src.forecasting.lstm_forecaster
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.environment.demand_generator import DemandGenerator
from src.environment.grid_env import load_config

WINDOW = 24
N_ZONES = 5
FEAT_DIM = N_ZONES + 4  # 5 demands + sin/cos hour + sin/cos day


class LSTMForecaster(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 64, n_zones: int = N_ZONES):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, n_zones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # predict from last hidden state


def generate_series(cfg: dict, n_weeks: int, seed: int) -> np.ndarray:
    """Continuous multi-week noisy demand series, shape (n_weeks*168, 5+4).

    Months are sampled per week (matching the env's episode-level season
    latent) so the forecaster sees the same distribution the agents face.
    """
    gen = DemandGenerator(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_weeks):
        month = int(rng.integers(1, 13))
        for h in range(168):
            demand = gen.sample(h, month, rng)
            hh, dd = h % 24, (h // 24) % 7
            time_enc = [
                np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24),
                np.sin(2 * np.pi * dd / 7), np.cos(2 * np.pi * dd / 7),
            ]
            rows.append(np.concatenate([demand, time_enc]))
    return np.array(rows, dtype=np.float32)


def make_windows(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(series) - WINDOW):
        xs.append(series[i : i + WINDOW])
        ys.append(series[i + WINDOW, :N_ZONES])
    return np.stack(xs), np.stack(ys)


def train(
    cfg: dict,
    n_weeks: int = 60,
    epochs: int = 15,
    seed: int = 42,
    out: str = "models/saved_models/lstm.pt",
) -> dict:
    torch.manual_seed(seed)
    # Normalize demand by per-zone base so the LSTM sees O(1) inputs.
    base = np.array([z["base_demand_mw"] for z in cfg["zones"]], dtype=np.float32)
    series = generate_series(cfg, n_weeks, seed)
    series[:, :N_ZONES] /= base
    x, y = make_windows(series)
    n_val = int(0.2 * len(x))
    x_tr, y_tr = torch.from_numpy(x[:-n_val]), torch.from_numpy(y[:-n_val])
    x_va, y_va = torch.from_numpy(x[-n_val:]), torch.from_numpy(y[-n_val:])

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(seed)
    batch = 128

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(x_tr))
        tr_loss = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_err_mw = ((model(x_va) - y_va).abs() * base).mean(dim=0)
        print(
            f"epoch {epoch + 1:2d}  train L1 {tr_loss / len(x_tr):.4f}  "
            f"val MAE per zone (MW): {[f'{v:.1f}' for v in val_err_mw]}",
            flush=True,
        )

    per_zone_mae = [float(v) for v in val_err_mw]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "base": base, "val_mae_mw": per_zone_mae}, out
    )
    print(f"saved {out}; mean val MAE {np.mean(per_zone_mae):.1f} MW")
    return {"val_mae_mw": per_zone_mae}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weeks", type=int, default=60)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/saved_models/lstm.pt")
    args = p.parse_args()
    train(load_config(), args.weeks, args.epochs, args.seed, args.out)
