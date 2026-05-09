# GraphFlow — Spatio-Temporal Traffic Speed Prediction with a T-GCN

A **Temporal Graph Convolutional Network** implemented from scratch in PyTorch — graph convolution + GRU recurrence — that predicts traffic speeds 15 minutes into the future across the **207-sensor METR-LA freeway network** (LA, USA), and beats an LSTM baseline that ignores road topology.

---

## The problem

Each of 207 sensors on Los Angeles freeways reports vehicle speed every 5 minutes. Given the last hour from all sensors, predict the next 15 minutes at every sensor.

A pure time-series model (LSTM) treats each sensor in isolation. But sensors are physically connected by roads — congestion at one propagates to its neighbours. **A graph neural network can learn this; an LSTM cannot.**

## Approach

| Model | Sees graph? | Sees history? |
|---|---|---|
| **T-GCN** | ✅ via graph convolution | ✅ via GRU |
| **LSTM** | ❌ | ✅ |

### T-GCN architecture

```
Input  (batch × 12 timesteps × 207 sensors)
  │
  for each timestep t:
      h ← GraphConv( x_t , Â , h )       # H' = σ(Â · H · W)
      h ← ReLU(h)
      h ← GRUCell( h_input , h )
  │
  Linear → 3-step prediction (next 15 min)
```

The graph convolution lets every sensor aggregate from its neighbours via the symmetrically-normalised adjacency matrix **Â = D⁻¹ᐟ² · A · D⁻¹ᐟ²**. The GRU then carries that neighbour-aware representation through time.

## Dataset

**Real graph topology:** the **METR-LA freeway sensor adjacency** (207 sensors, 1,313 edges) from the DCRNN paper (Li et al., ICLR 2018). Downloaded directly from the DCRNN repository.

**Synthetic signals on the real graph:** traffic speeds are generated using a neighbour-coupled dynamics model (daily rush-hour periodicity, neighbour-weighted state propagation, random congestion shocks). The canonical METR-LA signal loader had unresolved build dependencies in my environment, so I generated structurally-faithful signals on the real graph instead. This is honest — the graph is real LA infrastructure data; the signals are a controlled test.

- 207 sensors (real LA freeway network)
- 2,000 timesteps (~1 week of 5-min readings)
- 80 / 20 temporal train-test split
- Input window: 12 timesteps (1 hour) → output: 3 timesteps (15 min)

## Results

| Model | Test RMSE | Test MAE |
|---|---|---|
| **T-GCN** | **0.04658** | **0.03685** |
| LSTM (no graph) | 0.04781 | 0.03771 |

**T-GCN reduces RMSE by 2.6% and MAE by 2.3%** over the LSTM baseline. The graph convolution at each timestep gives the model access to neighbour states the LSTM physically cannot see.

The 2.6% margin is modest — but realistic. The METR-LA graph is dense (~13 neighbours per sensor on average), so an LSTM watching one sensor's history already infers much of what the graph carries. On sparser graphs the margin grows.

### Per-sensor analysis

The Pearson correlation between **node degree** and **per-sensor prediction error** is **r = −0.096** — a weak negative trend, suggesting that well-connected sensors are predicted slightly better but connectivity alone is not the dominant factor. Specific neighbour patterns matter more than raw neighbour count, which motivates attention-based extensions (Graph Attention Networks) as the natural next step.

### Figures

| File | Shows |
|---|---|
| `00_graph_topology.png` | The 207-sensor METR-LA freeway network |
| `01_loss_curves.png` | Train/test loss curves, T-GCN vs LSTM |
| `02_sensor_prediction.png` | One sensor's 5-min-ahead prediction over 200 timesteps |
| `03_per_sensor_error.png` | Per-sensor mean absolute error across the network |
| `04_model_comparison.png` | T-GCN vs LSTM, RMSE and MAE side by side |
| `05_ablation_hidden_dim.png` | Ablation: T-GCN test RMSE vs hidden dimension (16, 32, 64) |
| `06_degree_vs_error.png` | Node degree vs prediction error (Pearson scatter, r = −0.096) |

## How to run

```bash
pip install torch numpy matplotlib networkx
python graphflow.py
```

The METR-LA adjacency matrix is downloaded automatically from the DCRNN repository on first run. End-to-end runtime ≈ 20 min on Apple Silicon CPU (no GPU needed).

## Limitations

- **Synthetic signal dynamics, not real-world traffic.** The generator captures shape (periodicity, neighbour propagation, shocks) but not real-world irregularities (accidents, weather, lane closures). Future work: re-run on actual METR-LA signal data.
- **No attention mechanism.** A Graph Attention Network would let the model learn *which* neighbours matter for a given prediction — the natural next step given the weak degree-error correlation observed.
- **Single horizon.** I predict 15 minutes ahead; production traffic systems also predict 30/60-minute horizons jointly.
- **No external regressors.** Real prediction would condition on time-of-day, day-of-week, holidays, weather. Out of scope here.

## Stack
PyTorch · NumPy · NetworkX · Matplotlib

## References
- Zhao et al., *T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction*, IEEE T-ITS 2020.
- Li et al., *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018. (METR-LA dataset)
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.

