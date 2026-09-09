"""Recurrent sequence model (LSTM or GRU) with a direct multi-horizon head.

Input per time step: standardised log activity plus cyclical calendar encodings
(sin/cos hour, sin/cos weekday, weekend and holiday flags). The final hidden state is
concatenated with a learned cell embedding and the calendar encoding of every target
time, and a small MLP emits all horizons at once (direct strategy, no recursion).

Training: Adam, l1 loss (same rationale as LightGBM), gradient clipping, a plateau
learning-rate schedule and early stopping on the last week of the fitting window; the
weights of the best epoch are restored. Seeds are fixed for reproducibility.
"""

from __future__ import annotations

import copy
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..datasets import ForecastData, LogScaler, predictions_frame, targets_matrix
from ._common import eval_origins, fit_origins, holdout_split

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "cell": "lstm",
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "input_window": 168,
    "batch_size": 128,
    "max_epochs": 80,
    "patience": 10,
    "embedding_dim": 4,
    "clip_norm": 1.0,
}

CAL_DIM = 6


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calendar_encoding(calendar: pd.DataFrame) -> np.ndarray:
    """Cyclical encoding of the calendar flags: (n, 6) float32."""
    hour = calendar["hour"].to_numpy(dtype=np.float32)
    dow = calendar["dow"].to_numpy(dtype=np.float32)
    return np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                            calendar["is_weekend"].to_numpy(dtype=np.float32),
                            calendar["is_holiday"].to_numpy(dtype=np.float32)]).astype(np.float32)


def build_windows(data: ForecastData, scaler: LogScaler, origins: pd.DatetimeIndex, window: int):
    """Tensors (seq, target_cal, series_id, y) for every (series, origin) pair."""
    cal = calendar_encoding(data.calendar)
    horizons = np.asarray(data.horizons)
    seqs, tcals, ids, ys = [], [], [], []
    for sid, (name, s) in enumerate(data.series.items()):
        z = scaler.transform(name, s).astype(np.float32)
        pos = s.index.get_indexer(origins)
        # (n, window) index matrix of the input steps, then gather
        steps = pos[:, None] - np.arange(window - 1, -1, -1)[None, :]
        seq = np.concatenate([z[steps][..., None], cal[steps]], axis=-1)
        tcal = cal[pos[:, None] + horizons[None, :]].reshape(len(pos), -1)
        seqs.append(seq)
        tcals.append(tcal)
        ids.append(np.full(len(pos), sid, dtype=np.int64))
        ys.append(z[pos[:, None] + horizons[None, :]])
    return (torch.from_numpy(np.concatenate(seqs)), torch.from_numpy(np.concatenate(tcals)),
            torch.from_numpy(np.concatenate(ids)), torch.from_numpy(np.concatenate(ys)))


class SeqForecaster(nn.Module):
    def __init__(self, cell: str, n_series: int, n_horizons: int, hidden: int, layers: int, dropout: float,
                 emb_dim: int):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(1 + CAL_DIM, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.embed = nn.Embedding(n_series, emb_dim)
        head_in = hidden + emb_dim + n_horizons * CAL_DIM
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_horizons))

    def forward(self, seq, tcal, sid):
        out, _ = self.rnn(seq)
        last = out[:, -1, :]
        return self.head(torch.cat([last, self.embed(sid), tcal], dim=-1))


def _predict(model: nn.Module, tensors, batch: int, device) -> np.ndarray:
    model.eval()
    seq, tcal, sid, _ = tensors
    outs = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            outs.append(model(seq[i:i + batch].to(device), tcal[i:i + batch].to(device),
                              sid[i:i + batch].to(device)).cpu())
    return torch.cat(outs).numpy()


def _train(model: nn.Module, train_t, ho_t, p: dict, device) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.L1Loss()
    seq, tcal, sid, y = (t.to(device) for t in train_t)
    n, bs = len(seq), int(p["batch_size"])
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(int(p["max_epochs"])):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(seq[b], tcal[b], sid[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(p["clip_norm"]))
            opt.step()
            total += loss.item() * len(b)
        ho_loss = float(np.mean(np.abs(_predict(model, ho_t, bs, device) - ho_t[3].numpy())))
        sched.step(ho_loss)
        history.append({"epoch": epoch + 1, "train_l1": total / n, "holdout_l1": ho_loss})
        if ho_loss < best - 1e-5:
            best, best_state, bad = ho_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(p["patience"]):
                break
    model.load_state_dict(best_state)
    return {"epochs": len(history), "best_holdout_l1": best, "history": history}


def forecast(data: ForecastData, part: str, params: dict, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    p = {**DEFAULT_PARAMS, **params}
    window = int(p["input_window"])
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = LogScaler().fit(data)
    names = list(data.series)

    all_fit = fit_origins(data, part, min_history=window)
    inner_fit, holdout = holdout_split(all_fit)
    evaluation = eval_origins(data, part, min_history=window)

    model = SeqForecaster(p["cell"], len(names), len(data.horizons), int(p["hidden_size"]), int(p["num_layers"]),
                          float(p["dropout"]), int(p["embedding_dim"])).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    train_t = build_windows(data, scaler, inner_fit, window)
    ho_t = build_windows(data, scaler, holdout, window)
    info = _train(model, train_t, ho_t, p, device)
    info.update({"n_params": int(n_params), "device": str(device), "n_train_windows": int(len(train_t[0]))})
    log.info("%s: %d params, %d epochs, best holdout l1 %.4f", p["cell"], n_params, info["epochs"], info["best_holdout_l1"])

    ev_t = build_windows(data, scaler, evaluation, window)
    z_hat = _predict(model, ev_t, int(p["batch_size"]), device)
    sids = ev_t[2].numpy()
    frames = []
    for sid, n in enumerate(names):
        y_pred = scaler.inverse(n, z_hat[sids == sid])
        frames.append(predictions_frame(p["cell"], n, evaluation, data.horizons,
                                        targets_matrix(data.series[n], evaluation, data.horizons), y_pred))
    return pd.concat(frames, ignore_index=True), info
