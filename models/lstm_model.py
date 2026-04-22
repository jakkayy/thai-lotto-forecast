"""LSTM model — จับ sequential pattern จาก history"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from models.base_model import BaseLotteryModel

_SEQ_LEN = 20  # ใช้ 20 งวดล่าสุดเป็น input sequence


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(BaseLotteryModel):
    """
    LSTM รับ sequence ของ features ย้อนหลัง _SEQ_LEN งวด
    สำหรับแต่ละ candidate และ predict ว่าจะออกงวดถัดไปไหม
    """
    name = "lstm"

    def __init__(self, seq_len: int = _SEQ_LEN, epochs: int = 30, lr: float = 1e-3):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self._net: _LSTMNet | None = None
        self._feature_cols: list[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """สร้าง (X, y) sequences จาก DataFrame ที่เรียงตาม draw_date"""
        feat_cols = self._feature_cols
        X_list, y_list = [], []

        candidates = df["candidate"].unique()
        for cand in candidates:
            cand_df = df[df["candidate"] == cand].sort_values("draw_date")
            feats = cand_df[feat_cols].values.astype(np.float32)
            labels = cand_df["is_winner"].values.astype(np.float32)

            for i in range(self.seq_len, len(feats)):
                X_list.append(feats[i - self.seq_len:i])
                y_list.append(labels[i])

        if not X_list:
            return np.empty((0, self.seq_len, len(feat_cols))), np.empty(0)
        return np.array(X_list), np.array(y_list)

    def fit(self, df_train: pd.DataFrame) -> None:
        from features.engineer import FEATURE_COLS as BASE_COLS
        digit_cols = [c for c in df_train.columns if c.startswith("digit_")]
        self._feature_cols = [c for c in BASE_COLS + digit_cols if c in df_train.columns]

        X, y = self._build_sequences(df_train)
        if len(X) == 0:
            logger.warning("[lstm] not enough data to build sequences")
            return

        self._net = _LSTMNet(input_size=len(self._feature_cols)).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        pos = y.sum()
        neg = len(y) - pos
        pos_weight = torch.tensor([neg / pos] if pos > 0 else [1.0]).to(self._device)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X).to(self._device)
        y_t = torch.tensor(y).to(self._device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        self._net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(f"[lstm] epoch {epoch+1}/{self.epochs} loss={total_loss/len(loader):.4f}")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self._net is None:
            return pd.Series(np.zeros(len(df)), index=df.index)

        feat_cols = self._feature_cols
        scores = np.zeros(len(df))

        candidates = df["candidate"].unique()
        cand_to_idx = {c: df[df["candidate"] == c].index.tolist() for c in candidates}

        self._net.eval()
        with torch.no_grad():
            for cand in candidates:
                idxs = cand_to_idx[cand]
                cand_df = df.loc[idxs].sort_values("draw_date")
                feats = cand_df[feat_cols].values.astype(np.float32)

                if len(feats) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(feats), len(feat_cols)), dtype=np.float32)
                    feats = np.vstack([pad, feats])

                seq = feats[-self.seq_len:].reshape(1, self.seq_len, len(feat_cols))
                x_t = torch.tensor(seq).to(self._device)
                score = self._net(x_t).item()

                for idx in idxs:
                    scores[df.index.get_loc(idx)] = score

        return pd.Series(scores, index=df.index)

    def save(self, path: Path) -> None:
        if self._net:
            torch.save({"state_dict": self._net.state_dict(), "feature_cols": self._feature_cols}, str(path))

    def load(self, path: Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device)
        self._feature_cols = ckpt["feature_cols"]
        self._net = _LSTMNet(input_size=len(self._feature_cols)).to(self._device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
