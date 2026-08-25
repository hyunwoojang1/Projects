"""Module 1: Long-term TQQQ Entry Signal — ML/DL Ensemble.

Goal: "지금 TQQQ 사서 1년 들고 있으면 좋은 타이밍인가?"

Labels (1-year forward TQQQ return):
  3 → TQQQ 집중매수  (> +80%  : 역사적 폭락 바닥, 3배 레버리지 집중 진입)
  2 → QLD 매수       (+20~+80%: 상승 여력 있음, 2배 레버리지 적정)
  1 → QQQ 보유       (0~+20%  : 횡보·약상승, 레버리지 자제)
  0 → 현금·채권      (< 0%    : 하락 위험, TQQQ/QLD 전량 회피 → TLT·GLD 헤지)

Data: 30년 (1996~현재). TQQQ 2010 이전은 QQQ×3 합성.
CV:   5-fold anchored walk-forward, 252-day purge at boundary.
"""

from __future__ import annotations

import warnings
from datetime import datetime as _dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("./models")
SEQUENCE_LEN = 60      # LSTM/Transformer lookback window
FORWARD_DAYS = 252     # 1-year forward label
N_CLASSES = 4
PURGE_DAYS = 252       # boundary purge to prevent label leakage

TQQQ_LAUNCH = pd.Timestamp("2010-02-09")

LABEL_MAP = {
    0: "현금·채권 보유 (TLT·GLD 헤지)",
    1: "QQQ 보유 (레버리지 자제)",
    2: "QLD 매수 (2배 레버리지)",
    3: "TQQQ 집중매수 (3배 레버리지)",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def train(market: pd.DataFrame, fred: pd.DataFrame,
          factors: pd.DataFrame) -> dict:
    """5-fold CV + production model. Saves bundle. Returns CV summary."""
    ds = _build_dataset(market, fred, factors)
    if ds is None:
        return {"error": "데이터 부족 — 최소 500일 필요"}

    X, y, dates, tqqq_s, qqq_s = ds
    if len(X) < 500:
        return {"error": f"샘플 부족: {len(X)}개"}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5-fold walk-forward CV (fast mode) ────────────────────────────────────
    splits = _walk_forward_splits(len(X), n_folds=5)
    cv_results = []
    for fold_i, (train_end, test_start, test_end) in enumerate(splits):
        fold_b = _train_fold(X[:train_end], y[:train_end], fast=True)
        preds = np.argmax(_ensemble_predict_proba(X[test_start:test_end], fold_b), axis=1)
        acc = float(np.mean(preds == y[test_start:test_end]))
        cv_results.append({
            "fold": fold_i + 1,
            "train": f"{dates[0].date()} ~ {dates[train_end-1].date()}",
            "test":  f"{dates[test_start].date()} ~ {dates[test_end-1].date()}",
            "accuracy": round(acc, 4),
            "n_train": train_end,
            "n_test": test_end - test_start,
        })

    # ── Production model (all data, full HPO) ─────────────────────────────────
    prod = _train_fold(X, y, fast=False)
    prod.update({
        "X_all":        X,
        "dates_all":    dates,
        "tqqq_series":  tqqq_s,
        "qqq_series":   qqq_s,
        "market_vix":   market["^VIX"].ffill() if "^VIX" in market.columns else pd.Series(dtype=float),
        "feature_names": _feature_names(market, fred, factors),
    })

    fname = f"model_{_dt.now().strftime('%Y%m%d')}.pkl"
    joblib.dump(prod, MODEL_DIR / fname)

    dist = {LABEL_MAP[i]: int(np.sum(y == i)) for i in range(N_CLASSES)}
    return {
        "cv_results": cv_results,
        "cv_mean_accuracy": round(float(np.mean([r["accuracy"] for r in cv_results])), 4),
        "n_total_samples": len(X),
        "label_distribution": dist,
        "saved_model": fname,
    }


def predict(market: pd.DataFrame, fred: pd.DataFrame,
            factors: pd.DataFrame) -> dict:
    """Daily inference: signal + historical analog table."""
    model_files = sorted(MODEL_DIR.glob("model_*.pkl"))
    if not model_files:
        return train(market, fred, factors)
    bundle = joblib.load(model_files[-1])

    ds = _build_dataset(market, fred, factors)
    if ds is None:
        return {"error": "피처 생성 실패"}

    X, _, dates, tqqq_s, qqq_s = ds
    today_x = X[-1:]

    probs = _ensemble_predict_proba(today_x, bundle)
    pred_cls = int(np.argmax(probs[0]))
    conf = float(probs[0][pred_cls])

    analogs = _find_analogs(
        today_x,
        bundle.get("X_all"),
        bundle.get("dates_all"),
        bundle.get("tqqq_series"),
        bundle.get("qqq_series"),
        bundle.get("market_vix"),
    )

    valid_1y = [a["tqqq_1y_pct"] for a in analogs if a["tqqq_1y_pct"] is not None]
    valid_3y = [a["tqqq_3y_pct"] for a in analogs if a["tqqq_3y_pct"] is not None]

    expected: dict = {}
    if valid_1y:
        expected["tqqq_1y_median_pct"] = round(float(np.median(valid_1y)), 1)
        expected["tqqq_1y_range_pct"]  = [round(float(np.percentile(valid_1y, 25)), 1),
                                           round(float(np.percentile(valid_1y, 75)), 1)]
    if valid_3y:
        expected["tqqq_3y_median_pct"] = round(float(np.median(valid_3y)), 1)
        expected["tqqq_3y_range_pct"]  = [round(float(np.percentile(valid_3y, 25)), 1),
                                           round(float(np.percentile(valid_3y, 75)), 1)]

    return {
        "signal":        pred_cls,
        "signal_label":  LABEL_MAP.get(pred_cls, "알 수 없음"),
        "confidence":    round(conf, 4),
        "low_confidence": conf < 0.55,
        "probabilities": {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs[0])},
        "models": {
            "xgboost":     LABEL_MAP.get(_tree_predict(bundle.get("xgb"), today_x), "?"),
            "lightgbm":    LABEL_MAP.get(_tree_predict(bundle.get("lgb"), today_x), "?"),
            "lstm":        LABEL_MAP.get(_seq_predict(bundle.get("lstm"), today_x, bundle.get("scaler_lstm")), "?"),
            "transformer": LABEL_MAP.get(_seq_predict(bundle.get("transformer"), today_x, bundle.get("scaler_tf")), "?"),
        },
        "expected_returns": expected,
        "analogs":       analogs,
        "top_features":  _top_features(bundle),
    }


# ── Dataset builder ────────────────────────────────────────────────────────────

def _build_dataset(market: pd.DataFrame, fred: pd.DataFrame,
                   factors: pd.DataFrame):
    """Returns (X, y, dates, tqqq_series, qqq_series) or None."""
    try:
        if "QQQ" not in market.columns:
            return None

        tqqq_s = _synthetic_tqqq(market)
        qqq_s  = market["QQQ"].ffill()

        df = _compute_features(market, fred, factors)
        df = df.dropna()
        if len(df) < FORWARD_DAYS + 100:
            return None

        # 1-year forward TQQQ return → label
        tqqq_aligned = tqqq_s.reindex(df.index).ffill()
        fwd_ret = tqqq_aligned.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
        df["label"] = fwd_ret.map(_label_from_tqqq_return)
        df = df.dropna(subset=["label"])
        df = df[df["label"] >= 0]

        feat_cols = [c for c in df.columns if c != "label"]
        X     = df[feat_cols].values.astype(np.float32)
        y     = df["label"].values.astype(np.int32)
        dates = df.index
        return X, y, dates, tqqq_s, qqq_s

    except Exception as e:
        warnings.warn(f"_build_dataset failed: {e}")
        return None


def _synthetic_tqqq(market: pd.DataFrame) -> pd.Series:
    """TQQQ price series. Pre-launch (before 2010-02-09): QQQ×3 합성."""
    qqq = market["QQQ"].ffill()

    if "TQQQ" not in market.columns:
        return (1 + qqq.pct_change().fillna(0) * 3).cumprod() * 100

    tqqq_real = market["TQQQ"].ffill()
    launch = tqqq_real.first_valid_index()
    if launch is None:
        return (1 + qqq.pct_change().fillna(0) * 3).cumprod() * 100

    pre = qqq[qqq.index < launch]
    if pre.empty:
        return tqqq_real

    pre_cum = (1 + pre.pct_change().fillna(0) * 3).cumprod()
    scale   = tqqq_real[launch] / pre_cum.iloc[-1]
    return pd.concat([pre_cum * scale, tqqq_real[tqqq_real.index >= launch]]).ffill()


def _label_from_tqqq_return(ret: float) -> int:
    if pd.isna(ret):
        return -1
    if ret > 0.80:
        return 3   # TQQQ 집중매수: 1년 후 +80% 초과
    if ret > 0.20:
        return 2   # QLD 매수: +20~+80%
    if ret >= 0.00:
        return 1   # QQQ 보유: 0~+20%
    return 0       # 현금·채권: 손실 구간


# ── Feature engineering ────────────────────────────────────────────────────────

def _compute_features(market: pd.DataFrame, fred: pd.DataFrame,
                      factors: pd.DataFrame) -> pd.DataFrame:
    df = _compute_technical_features(market)
    df = _merge_macro_features(df, market, fred)
    df = _merge_wrds_features(df, factors)
    return df


def _compute_technical_features(market: pd.DataFrame) -> pd.DataFrame:
    if "QQQ" not in market.columns:
        return pd.DataFrame()

    df  = pd.DataFrame(index=market.index)
    qqq = market["QQQ"].ffill()

    # RSI
    for p in [7, 14, 21]:
        df[f"rsi_{p}"] = _rsi(qqq, p)

    # MA deviations
    for w in [50, 100, 200]:
        ma = qqq.rolling(w).mean()
        df[f"ma_dev_{w}"] = (qqq - ma) / ma

    # MACD
    ema12 = qqq.ewm(span=12).mean()
    ema26 = qqq.ewm(span=26).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger %B
    bb_mid = qqq.rolling(20).mean()
    bb_std = qqq.rolling(20).std()
    df["bb_pct_b"] = (qqq - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ATH drawdown — 핵심 "폭락 감지" 피처
    ath = qqq.expanding().max()
    df["ath_drawdown"] = (qqq - ath) / ath

    # Momentum (multi-horizon)
    df["qqq_3m_return"]  = qqq.pct_change(63)
    df["qqq_6m_return"]  = qqq.pct_change(126)
    df["qqq_12m_return"] = qqq.pct_change(252)

    # ATR (normalised)
    df["atr_norm"] = qqq.rolling(14).std() / qqq

    # Volume features
    vol_col = "QQQ_vol"
    if vol_col in market.columns:
        vol = market[vol_col].ffill()
        df["vol_ratio_5d"]  = vol.rolling(5).mean()  / vol.rolling(20).mean()
        df["vol_ratio_20d"] = vol.rolling(20).mean() / vol.rolling(60).mean()

    # VIX — level, percentile rank, change
    if "^VIX" in market.columns:
        vix = market["^VIX"].ffill()
        df["vix_level"]    = vix
        df["vix_pct_rank"] = vix.rolling(252, min_periods=60).rank(pct=True)
        df["vix_chg_20d"]  = vix.pct_change(20)

    # TLT / SPY ratio
    if "TLT" in market.columns and "SPY" in market.columns:
        df["tlt_spy_ratio"]     = market["TLT"] / market["SPY"]
        df["tlt_spy_ratio_chg"] = df["tlt_spy_ratio"].pct_change(20)

    # DXY
    if "DX-Y.NYB" in market.columns:
        df["dxy_chg_20d"] = market["DX-Y.NYB"].ffill().pct_change(20)

    # HYG (credit stress)
    if "HYG" in market.columns:
        hyg = market["HYG"].ffill()
        df["hyg_chg_20d"] = hyg.pct_change(20)

    return df


def _merge_macro_features(df: pd.DataFrame, market: pd.DataFrame,
                           fred: pd.DataFrame) -> pd.DataFrame:
    if fred.empty:
        return df

    if "DGS10" in fred.columns:
        df["rate_10y"] = fred["DGS10"].reindex(df.index, method="ffill")

    if "SPREAD_10Y2Y" in fred.columns:
        df["yield_spread"] = fred["SPREAD_10Y2Y"].reindex(df.index, method="ffill")

    if "M2SL" in fred.columns:
        m2 = fred["M2SL"].reindex(df.index, method="ffill")
        df["m2_growth"] = m2.pct_change(252)

    if "CPIAUCSL" in fred.columns:
        cpi = fred["CPIAUCSL"].reindex(df.index, method="ffill")
        df["cpi_yoy"] = cpi.pct_change(252) * 100

    if "UNRATE" in fred.columns:
        df["unrate"] = fred["UNRATE"].reindex(df.index, method="ffill")

    return df


def _merge_wrds_features(df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    if factors.empty:
        df["earnings_surprise"] = 0.0
        df["quality_factor"]    = 0.0
        df["wrds_available"]    = 0.0
        return df

    monthly = (factors.groupby("datadate")
                      .agg({"earnings_surprise": "mean", "gpa": "mean"})
                      .rename(columns={"gpa": "quality_factor"}))
    monthly.index = pd.to_datetime(monthly.index)
    aligned = monthly.reindex(df.index, method="ffill")
    df["earnings_surprise"] = aligned["earnings_surprise"].fillna(0)
    df["quality_factor"]    = aligned["quality_factor"].fillna(0)
    df["wrds_available"]    = (~aligned["earnings_surprise"].isna()).astype(float)
    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _feature_names(market, fred, factors) -> list[str]:
    df = _compute_features(market, fred, factors).dropna()
    return [c for c in df.columns if c != "label"]


# ── Walk-forward splits ────────────────────────────────────────────────────────

def _walk_forward_splits(n: int, n_folds: int = 5) -> list[tuple]:
    """Anchored walk-forward. Each fold: train=[0:train_end], test=[test_start:test_end].
    252-day purge zone between train_end and test_start."""
    min_train = max(500, int(n * 0.40))
    available = n - min_train
    fold_size = max(100, available // n_folds)

    splits = []
    for i in range(n_folds):
        train_end  = min_train + i * fold_size
        test_start = train_end + PURGE_DAYS
        test_end   = min(test_start + fold_size, n)
        if test_start >= n or (test_end - test_start) < 50:
            break
        splits.append((train_end, test_start, test_end))
    return splits


# ── Training ───────────────────────────────────────────────────────────────────

def _train_fold(X: np.ndarray, y: np.ndarray, fast: bool = False) -> dict:
    """Train XGB + LGB + LSTM + Transformer + Meta on (X, y)."""
    n_trials = 10 if fast else 30
    epochs   = 20 if fast else 50

    # 80% base / 20% meta
    split = max(100, int(len(X) * 0.8))
    X_b, y_b = X[:split], y[:split]
    X_m, y_m = X[split:], y[split:]

    xgb_m           = _optimize_xgb(X_b, y_b, n_trials)
    lgb_m           = _optimize_lgb(X_b, y_b, n_trials)
    lstm_m, sc_lstm = _train_lstm_model(X_b, y_b, epochs)
    tf_m,   sc_tf   = _train_transformer_model(X_b, y_b, epochs)

    bundle: dict = {
        "xgb": xgb_m, "lgb": lgb_m,
        "lstm": lstm_m, "scaler_lstm": sc_lstm,
        "transformer": tf_m, "scaler_tf": sc_tf,
    }

    if len(X_m) >= 10 and len(np.unique(y_m)) >= 2:
        meta_x = _collect_meta_features(X_m, xgb_m, lgb_m, lstm_m, sc_lstm, tf_m, sc_tf)
        meta = LogisticRegression(max_iter=500, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta.fit(meta_x, y_m)
        bundle["meta"] = meta

    return bundle


def _optimize_xgb(X: np.ndarray, y: np.ndarray, n_trials: int = 30):
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = int(len(X) * 4 / 5)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "mlogloss", "random_state": 42,
            "num_class": N_CLASSES, "objective": "multi:softprob",
            "use_label_encoder": False,
        }
        m = XGBClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[:split], y[:split])
        return float(np.mean(m.predict(X[split:]) == y[split:]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    model = XGBClassifier(
        **bp, eval_metric="mlogloss", random_state=42,
        objective="multi:softprob", num_class=N_CLASSES,
        use_label_encoder=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _optimize_lgb(X: np.ndarray, y: np.ndarray, n_trials: int = 30):
    import optuna
    from lightgbm import LGBMClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = int(len(X) * 4 / 5)

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "max_depth":     trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 15, 100),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42, "verbose": -1,
        }
        m = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[:split], y[:split])
        return float(np.mean(m.predict(X[split:]) == y[split:]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    model = LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _train_lstm_model(X: np.ndarray, y: np.ndarray, epochs: int = 50) -> tuple:
    try:
        import torch  # noqa: F401
    except ImportError:
        warnings.warn("PyTorch not installed — LSTM skipped")
        return None, None

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_seq, y_seq = _make_sequences(X_sc, y)
    if len(X_seq) < 50:
        return None, scaler

    model = _LSTMModel(X.shape[1], hidden=128, n_classes=N_CLASSES)
    _fit_pytorch(model, X_seq, y_seq, epochs=epochs, lr=1e-3)
    return model, scaler


def _train_transformer_model(X: np.ndarray, y: np.ndarray, epochs: int = 50) -> tuple:
    try:
        import torch  # noqa: F401
    except ImportError:
        warnings.warn("PyTorch not installed — Transformer skipped")
        return None, None

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_seq, y_seq = _make_sequences(X_sc, y)
    if len(X_seq) < 50:
        return None, scaler

    model = _TransformerModel(X.shape[1], n_heads=4, n_classes=N_CLASSES)
    _fit_pytorch(model, X_seq, y_seq, epochs=epochs, lr=1e-3)
    return model, scaler


def _make_sequences(X: np.ndarray, y: np.ndarray) -> tuple:
    seqs, labels = [], []
    for i in range(SEQUENCE_LEN, len(X)):
        seqs.append(X[i - SEQUENCE_LEN:i])
        labels.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)


def _fit_pytorch(model, X_seq: np.ndarray, y_seq: np.ndarray,
                 epochs: int = 50, lr: float = 1e-3) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq)),
        batch_size=64, shuffle=True,
    )
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    for _ in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step(ep_loss)


# ── PyTorch model classes (module-level for joblib pickle) ────────────────────

try:
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden: int, n_classes: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True,
                                num_layers=2, dropout=0.3)
            self.fc   = nn.Linear(hidden, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class _TransformerNet(nn.Module):
        def __init__(self, n_features: int, n_heads: int, n_classes: int):
            super().__init__()
            d = max(n_features, 64)
            if d % n_heads != 0:
                d = n_heads * (d // n_heads + 1)
            self.proj = nn.Linear(n_features, d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(d)
            self.ffn  = nn.Sequential(
                nn.Linear(d, d * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d * 2, d)
            )
            self.fc = nn.Linear(d, n_classes)

        def forward(self, x):
            x = self.proj(x)
            a, _ = self.attn(x, x, x)
            x = self.norm(x + a)
            x = self.norm(x + self.ffn(x))
            return self.fc(x[:, -1, :])

except ImportError:
    _LSTMNet = None
    _TransformerNet = None


class _LSTMModel:
    def __init__(self, n_features: int, hidden: int = 128, n_classes: int = 3):
        if _LSTMNet is None:
            raise ImportError("PyTorch not installed")
        self.net = _LSTMNet(n_features, hidden, n_classes)
        self.n_features = n_features

    def to(self, device):
        self.net = self.net.to(device); self._device = device; return self

    def train(self): self.net.train()
    def eval(self):  self.net.eval()
    def __call__(self, x): return self.net(x)
    def parameters(self): return self.net.parameters()


class _TransformerModel:
    def __init__(self, n_features: int, n_heads: int = 4, n_classes: int = 3):
        if _TransformerNet is None:
            raise ImportError("PyTorch not installed")
        self.net = _TransformerNet(n_features, n_heads, n_classes)
        self.n_features = n_features

    def to(self, device):
        self.net = self.net.to(device); self._device = device; return self

    def train(self): self.net.train()
    def eval(self):  self.net.eval()
    def __call__(self, x): return self.net(x)
    def parameters(self): return self.net.parameters()


# ── Ensemble ───────────────────────────────────────────────────────────────────

def _collect_meta_features(X, xgb, lgb, lstm, sc_lstm, tf, sc_tf) -> np.ndarray:
    parts = []
    if xgb is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parts.append(xgb.predict_proba(X))
    if lgb is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parts.append(lgb.predict_proba(X))
    if lstm is not None and sc_lstm is not None:
        parts.append(_seq_proba(lstm, X, sc_lstm))
    if tf is not None and sc_tf is not None:
        parts.append(_seq_proba(tf, X, sc_tf))
    return np.hstack(parts) if parts else np.zeros((len(X), N_CLASSES))


def _ensemble_predict_proba(X: np.ndarray, bundle: dict) -> np.ndarray:
    meta_x = _collect_meta_features(
        X,
        bundle.get("xgb"), bundle.get("lgb"),
        bundle.get("lstm"), bundle.get("scaler_lstm"),
        bundle.get("transformer"), bundle.get("scaler_tf"),
    )
    meta = bundle.get("meta")
    if meta is None or len(meta_x) == 0:
        return np.full((1, N_CLASSES), 1.0 / N_CLASSES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return meta.predict_proba(meta_x)


# ── Analog finder ──────────────────────────────────────────────────────────────

def _find_analogs(today_x: np.ndarray, X_all, dates_all,
                  tqqq_s, qqq_s, vix_s, n: int = 10) -> list[dict]:
    """가장 유사한 과거 10개 구간 + 그 이후 TQQQ/QQQ 수익률."""
    if X_all is None or dates_all is None:
        return []
    try:
        scaler  = StandardScaler()
        X_norm  = scaler.fit_transform(X_all)
        t_norm  = scaler.transform(today_x)
        dists   = np.linalg.norm(X_norm - t_norm, axis=1)

        # 최근 FORWARD_DAYS는 미래 수익률 계산 불가 → 제외
        dists[-FORWARD_DAYS:] = np.inf

        results = []
        for i in np.argsort(dists)[:n]:
            date = dates_all[i]
            sim  = round(100 / (1 + dists[i]), 1)

            t1y = _fwd_return(tqqq_s, date, 252)
            t3y = _fwd_return(tqqq_s, date, 756)
            q1y = _fwd_return(qqq_s,  date, 252)
            q3y = _fwd_return(qqq_s,  date, 756)

            vix_val = None
            if vix_s is not None and date in vix_s.index:
                v = float(vix_s[date])
                vix_val = round(v, 1) if not np.isnan(v) else None

            dd = None
            if qqq_s is not None:
                hist = qqq_s[:date]
                if len(hist) > 0:
                    ath = float(hist.max())
                    cur = float(qqq_s.get(date, np.nan))
                    if ath > 0 and not np.isnan(cur):
                        dd = round((cur - ath) / ath * 100, 1)

            results.append({
                "date":               str(date.date()),
                "similarity_pct":     sim,
                "vix":                vix_val,
                "drawdown_from_ath_pct": dd,
                "tqqq_1y_pct": round(t1y * 100, 1) if t1y is not None else None,
                "tqqq_3y_pct": round(t3y * 100, 1) if t3y is not None else None,
                "qqq_1y_pct":  round(q1y * 100, 1) if q1y is not None else None,
                "qqq_3y_pct":  round(q3y * 100, 1) if q3y is not None else None,
            })

        results.sort(key=lambda x: x["similarity_pct"], reverse=True)
        return results

    except Exception as e:
        warnings.warn(f"_find_analogs failed: {e}")
        return []


def _fwd_return(series: pd.Series, date: pd.Timestamp, days: int):
    """date 이후 days일 수익률. 데이터 부족 시 None."""
    if series is None:
        return None
    try:
        idx   = series.index.searchsorted(date)
        end_i = idx + days
        if end_i >= len(series):
            return None
        p0 = float(series.iloc[idx])
        p1 = float(series.iloc[end_i])
        return (p1 - p0) / p0 if p0 > 0 else None
    except Exception:
        return None


# ── Helper functions ───────────────────────────────────────────────────────────

def _tree_predict(model, X: np.ndarray) -> int:
    if model is None:
        return 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(model.predict(X)[0])


def _seq_proba(model, X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_sc    = scaler.transform(X)

        if len(X_sc) < SEQUENCE_LEN:
            return np.full((len(X), N_CLASSES), 1.0 / N_CLASSES)

        seqs = []
        for i in range(len(X_sc)):
            s = max(0, i - SEQUENCE_LEN + 1)
            seq = X_sc[s:i + 1]
            if len(seq) < SEQUENCE_LEN:
                pad = np.zeros((SEQUENCE_LEN - len(seq), X_sc.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)

        t = torch.tensor(np.array(seqs, dtype=np.float32)).to(device)
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(t), dim=1).cpu().numpy()
        return probs
    except Exception:
        return np.full((len(X), N_CLASSES), 1.0 / N_CLASSES)


def _seq_predict(model, X: np.ndarray, scaler) -> int:
    if model is None or scaler is None:
        return 1
    return int(np.argmax(_seq_proba(model, X, scaler)[-1]))


def _top_features(bundle: dict) -> list[dict]:
    xgb   = bundle.get("xgb")
    names = bundle.get("feature_names", [])
    if xgb is None or not names:
        return []
    try:
        imp   = xgb.feature_importances_
        total = imp.sum()
        if total == 0:
            return []
        top = np.argsort(imp)[::-1][:7]
        return [{"name": names[i] if i < len(names) else f"feat_{i}",
                 "importance": round(float(imp[i] / total), 4)} for i in top]
    except Exception:
        return []
