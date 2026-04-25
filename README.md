# GMM MT5 Strategy

## Algorithm
Gaussian Mixture Model (unsupervised clustering)

- Detects market regimes
- Maps clusters to BUY / SELL / FLAT

## Run training

```text
python train_mt5_gmm_classifier.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon 8 --output output_gmm_XAGUSD_M15
python train_mt5_gmm_classifier.py --symbol BTCUSD --timeframe M15 --bars 80000 --horizon 8 --output output_gmm_BTCUSD_M15
```

## Files generated

- gmm.onnx
- mapping.json

## MT5

1. Copy ONNX + mapping.json
2. Load EA
3. Set threshold ~0.55–0.65

## Notes

- Works best for regime detection
- Combine with filters (ATR, trend)