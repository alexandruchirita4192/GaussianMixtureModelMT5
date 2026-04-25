from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "dist_sma_10",
    "dist_sma_20",
    "zscore_20",
    "atr_pct_14",
]

SELL_CLASS = -1
FLAT_CLASS = 0
BUY_CLASS = 1


# ================= DATA =================

def fetch_rates(symbol, timeframe, bars):
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    tf_map = {
        "M15": mt5.TIMEFRAME_M15,
    }

    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    mt5.shutdown()

    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def build_features(df, horizon):
    df = df.copy()

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    df["dist_sma_10"] = df["close"] / df["sma_10"] - 1
    df["dist_sma_20"] = df["close"] / df["sma_20"] - 1

    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct_14"] = df["atr_14"] / close

    df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1

    return df.dropna()


# ================= LABEL =================

def label_data(df, quantile=0.3):
    threshold = df["fwd_ret"].abs().quantile(quantile)

    df["target"] = 0
    df.loc[df["fwd_ret"] > threshold, "target"] = 1
    df.loc[df["fwd_ret"] < -threshold, "target"] = -1

    return df, threshold


# ================= MODEL =================

def train_gmm(X):
    model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=42,
        max_iter=200
    )
    model.fit(X)
    return model


def map_clusters_to_classes(df, clusters):
    df = df.copy()
    df["cluster"] = clusters

    mapping = {}
    for c in sorted(df["cluster"].unique()):
        c_int = int(c)

        mean_ret = df[df["cluster"] == c]["fwd_ret"].mean()

        if mean_ret > 0:
            mapping[c_int] = int(BUY_CLASS)
        elif mean_ret < 0:
            mapping[c_int] = int(SELL_CLASS)
        else:
            mapping[c_int] = int(FLAT_CLASS)

    return mapping


def predict_with_mapping(model, X, mapping):
    probs = model.predict_proba(X)
    clusters = probs.argmax(axis=1)

    pred = np.array([mapping[c] for c in clusters])
    conf = probs.max(axis=1)

    return pred, conf


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAGUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--bars", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--output", default="output_gmm")

    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)

    df = fetch_rates(args.symbol, args.timeframe, args.bars)
    df = build_features(df, args.horizon)
    df, barrier = label_data(df)

    split = int(len(df) * args.train_ratio)
    train, test = df[:split], df[split:]

    X_train = train[FEATURE_COLS].astype(np.float32)
    X_test = test[FEATURE_COLS].astype(np.float32)

    model = train_gmm(X_train)

    mapping = map_clusters_to_classes(train, model.predict(X_train))

    pred, conf = predict_with_mapping(model, X_test, mapping)

    acc = accuracy_score(test["target"], pred)

    print("Accuracy:", acc)

    # ================= ONNX =================
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, len(FEATURE_COLS)]))]
    )

    with open(f"{args.output}/gmm.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # save mapping
    with open(f"{args.output}/mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    main()