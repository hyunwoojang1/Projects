"""Module 2: Market Regime Classification — HMM + Dynamic Factor Weights (Kalman Filter).

References:
  - Ang & Bekaert (2002): regime-switching asset allocation
  - Guidolin & Timmermann (2007): multivariate regime switching
  - Novy-Marx (2013): quality factor (GP/A)
  - Asness, Moskowitz & Pedersen (2013): value & momentum everywhere

Pipeline:
  1. Gaussian HMM (3 states, BIC-optimized) on [SPY_ret, VIX, spread, TLT_ret]
  2. Kalman Filter tracks dynamic factor weights (momentum, value, quality, size)
  3. Regime → strategy mapping
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("./output/regime_model.pkl")

# Human-readable regime labels (assigned post-hoc by inspecting learned parameters)
REGIME_LABELS = {
    0: "강세 저변동 (Bull, Low Vol)",
    1: "약세 고변동 (Bear / Correction)",
    2: "전환 / 회복 (Recovery / Transition)",
}

REGIME_STRATEGIES = {
    0: "QQQ/QLD 유리, 성장주 ETF 비중 확대 권장",
    1: "방어적 포지션, TLT/GLD 헤지 고려, QQQ 비중 축소",
    2: "중립 포지션, 섹터 분산, 시그널 확인 후 진입",
}

FACTOR_NAMES = ["momentum", "value", "quality", "size"]


# ── Public API ────────────────────────────────────────────────────────────────

def train(market: pd.DataFrame, factor_returns: pd.DataFrame) -> dict:
    """Train HMM + Kalman filter. Saves model to disk."""
    obs = _build_observations(market)
    if obs is None or len(obs) < 60:
        return {"error": "데이터 부족"}

    n_states = _select_n_states(obs)
    hmm_model = _fit_hmm(obs, n_states)

    kalman_state = _init_kalman(factor_returns)

    model_bundle = {
        "hmm": hmm_model,
        "n_states": n_states,
        "obs_columns": ["spy_ret", "vix", "spread", "tlt_ret"],
        "kalman_state": kalman_state,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)

    return predict(market, factor_returns, model_bundle)


def predict(market: pd.DataFrame, factor_returns: pd.DataFrame,
            model_bundle: dict | None = None) -> dict:
    """Run daily regime prediction. Loads saved model if bundle not provided."""
    if model_bundle is None:
        if not MODEL_PATH.exists():
            return train(market, factor_returns)
        model_bundle = joblib.load(MODEL_PATH)

    obs = _build_observations(market)
    if obs is None or len(obs) < 10:
        return {"error": "관측 데이터 부족"}

    hmm = model_bundle["hmm"]
    n_states = model_bundle["n_states"]

    # Posterior probabilities for today
    log_prob, posteriors = hmm.score_samples(obs)
    current_probs = posteriors[-1]

    # Most likely current regime
    current_state = int(np.argmax(current_probs))
    # Reorder states: sort by mean SPY return (ascending) so state 0 = bull
    current_state = _remap_state(hmm, current_state, n_states)
    remapped_probs = _remap_probs(hmm, current_probs, n_states)

    # Dynamic factor weights (Kalman filter update)
    factor_weights = _update_kalman(factor_returns, model_bundle["kalman_state"])
    dominant_factor = max(factor_weights, key=factor_weights.get)

    return {
        "current_state": current_state,
        "state_label": REGIME_LABELS.get(current_state, f"State {current_state}"),
        "state_probs": {
            REGIME_LABELS.get(i, f"State {i}"): round(float(p), 4)
            for i, p in enumerate(remapped_probs)
        },
        "strategy": REGIME_STRATEGIES.get(current_state, "알 수 없음"),
        "factor_weights": {k: round(v, 4) for k, v in factor_weights.items()},
        "dominant_factor": dominant_factor,
        "regime_description": _describe(current_state, dominant_factor),
    }


# ── Observation matrix ────────────────────────────────────────────────────────

def _build_observations(market: pd.DataFrame) -> np.ndarray | None:
    needed = ["SPY", "^VIX", "TLT"]
    if not all(c in market.columns for c in needed):
        return None

    spy_ret = market["SPY"].pct_change().fillna(0)
    vix = market["^VIX"].ffill().fillna(20.0)
    tlt_ret = market["TLT"].pct_change().fillna(0)

    # Yield spread proxy: TLT price change as substitute for rate spread direction
    # (TLT falls when rates rise, so inverted sign ≈ spread)
    spread = -tlt_ret.rolling(5).mean().fillna(0)

    df = pd.DataFrame({
        "spy_ret": spy_ret,
        "vix": (vix - vix.mean()) / (vix.std() + 1e-8),
        "spread": (spread - spread.mean()) / (spread.std() + 1e-8),
        "tlt_ret": tlt_ret,
    }).dropna()

    return df.values


# ── HMM ──────────────────────────────────────────────────────────────────────

def _select_n_states(obs: np.ndarray) -> int:
    """BIC-based optimal state selection (search 2–5)."""
    try:
        from hmmlearn.hmm import GaussianHMM
        best_bic = np.inf
        best_n = 3

        for n in range(2, 6):
            try:
                model = GaussianHMM(n_components=n, covariance_type="full",
                                    n_iter=200, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(obs)
                log_l = model.score(obs)
                n_params = n * n + n * obs.shape[1] + n * obs.shape[1] ** 2
                bic = -2 * log_l + n_params * np.log(len(obs))
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception:
                continue

        return best_n
    except ImportError:
        warnings.warn("hmmlearn not installed — using 3 states")
        return 3


def _fit_hmm(obs: np.ndarray, n_states: int):
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=500, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(obs)
    return model


def _remap_state(hmm, state: int, n_states: int) -> int:
    """Map HMM states so state 0 = highest mean SPY return (bull)."""
    mean_returns = [float(hmm.means_[s][0]) for s in range(n_states)]
    order = sorted(range(n_states), key=lambda s: mean_returns[s], reverse=True)
    pos = order.index(state)
    return min(pos, 2)  # cap at 2 for our 3-label system


def _remap_probs(hmm, probs: np.ndarray, n_states: int) -> list[float]:
    mean_returns = [float(hmm.means_[s][0]) for s in range(n_states)]
    order = sorted(range(n_states), key=lambda s: mean_returns[s], reverse=True)
    remapped = [0.0, 0.0, 0.0]
    for rank, orig_state in enumerate(order):
        if rank < 3:
            remapped[rank] += float(probs[orig_state])
    return remapped


# ── Kalman Filter for dynamic factor weights ──────────────────────────────────

def _init_kalman(factor_returns: pd.DataFrame) -> dict:
    """Initialize Kalman filter state for factor weight tracking."""
    n = len(FACTOR_NAMES)
    return {
        "x": np.ones(n) / n,         # initial equal-weight state
        "P": np.eye(n) * 0.1,        # state covariance
        "Q": np.eye(n) * 0.01,       # process noise
        "R": np.eye(n) * 0.05,       # observation noise
        "last_update": None,
    }


def _update_kalman(factor_returns: pd.DataFrame, state: dict) -> dict[str, float]:
    """Run Kalman filter update step using recent factor performance."""
    n = len(FACTOR_NAMES)

    if factor_returns.empty:
        weights = np.abs(state["x"])
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return dict(zip(FACTOR_NAMES, weights.tolist()))

    # Observation: recent factor return magnitude as proxy for "factor is working"
    obs = np.zeros(n)
    for i, fname in enumerate(FACTOR_NAMES):
        if fname in factor_returns.columns:
            recent = factor_returns[fname].dropna().tail(20)
            if not recent.empty:
                obs[i] = float(recent.abs().mean())

    # Predict step
    x = state["x"]
    P = state["P"]
    Q = state["Q"]
    R = state["R"]

    x_pred = x
    P_pred = P + Q

    # Update step (H = identity: we observe the state directly)
    H = np.eye(n)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ (obs - H @ x_pred)
    P_new = (np.eye(n) - K @ H) @ P_pred

    state["x"] = x_new
    state["P"] = P_new

    # Normalize to sum to 1
    weights = np.abs(x_new)
    total = weights.sum()
    if total > 1e-8:
        weights = weights / total

    return dict(zip(FACTOR_NAMES, weights.tolist()))


def _describe(state: int, dominant_factor: str) -> str:
    factor_map = {
        "momentum": "모멘텀",
        "value": "밸류",
        "quality": "퀄리티",
        "size": "사이즈",
    }
    factor_kr = factor_map.get(dominant_factor, dominant_factor)
    regime_kr = REGIME_LABELS.get(state, "알 수 없음")
    return f"현재 시장: {regime_kr} | {factor_kr} 팩터 주도"
