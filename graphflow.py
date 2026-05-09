"""
GraphFlow — Spatio-Temporal Traffic Prediction on the METR-LA Freeway Network
T-GCN vs LSTM baseline — implemented from scratch in PyTorch.
Real graph topology (207 LA freeway sensors from DCRNN paper) + synthetic neighbor-coupled signals.
"""
import os
import pickle
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

torch.manual_seed(42); np.random.seed(42)

# =============================================
# 1. REAL GRAPH (METR-LA) + SYNTHETIC SIGNALS
# =============================================
def load_metrla_adjacency():
    """Download the real METR-LA adjacency from the DCRNN paper repo."""
    url = "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl"
    fname = "metr_la_adj.pkl"
    if not os.path.exists(fname):
        print("Downloading METR-LA adjacency matrix (207 LA freeway sensors)...")
        urllib.request.urlretrieve(url, fname)
        print(f"Saved to {fname}")
    with open(fname, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    # adj_mx is a Gaussian-thresholded distance kernel; threshold to binary topology
    A = (adj_mx > 0.1).astype(float)
    A = ((A + A.T) > 0).astype(float)            # ensure symmetric
    np.fill_diagonal(A, 1)                        # self-loops
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A, A_norm

def generate_traffic_signals(A, T=2000, seed=42):
    """Each node's next state explicitly depends on its NEIGHBORS' present states.
    Forces graph-aware models to outperform per-node baselines."""
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    A_row = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    base = rng.uniform(0.6, 0.8, n)
    t_arr = np.arange(T)
    daily = 0.12 * np.sin(2*np.pi*t_arr/288)

    signals = np.zeros((T, n))
    signals[0] = base + rng.randn(n) * 0.04
    for t in range(1, T):
        own = 0.55 * signals[t-1]
        neighbour = 0.35 * (signals[t-1] @ A_row.T)        # graph dependency
        forcing = 0.10 * (base + daily[t])
        noise = rng.randn(n) * 0.025
        signals[t] = own + neighbour + forcing + noise
        # occasional congestion shocks that propagate via the graph
        if rng.rand() < 0.08:
            signals[t, rng.randint(n)] -= 0.25

    signals = (signals - signals.min()) / (signals.max() - signals.min() + 1e-8)
    return signals.astype(np.float32)

NUM_TIMESTEPS = 2000
print("Loading real METR-LA adjacency (207 LA freeway sensors)...")
A_raw, A_norm = load_metrla_adjacency()
NUM_NODES = A_raw.shape[0]
print(f"Graph: {NUM_NODES} nodes, {int((A_raw.sum() - NUM_NODES)//2)} edges")

print("Generating synthetic neighbour-coupled traffic signals on real topology...")
signals = generate_traffic_signals(A_raw, NUM_TIMESTEPS)
A_norm_t = torch.FloatTensor(A_norm)
print(f"Signals: {signals.shape}")

# =============================================
# 2. WINDOWED DATASET (1-hour input → 15-min prediction)
# =============================================
WIN_IN, WIN_OUT = 12, 3
def make_windows(s, wi, wo):
    X, Y = [], []
    for t in range(len(s) - wi - wo):
        X.append(s[t:t+wi]); Y.append(s[t+wi:t+wi+wo])
    return np.stack(X), np.stack(Y)

X, Y = make_windows(signals, WIN_IN, WIN_OUT)
split = int(0.8 * len(X))
X_tr, Y_tr = torch.FloatTensor(X[:split]), torch.FloatTensor(Y[:split])
X_te, Y_te = torch.FloatTensor(X[split:]), torch.FloatTensor(Y[split:])
print(f"Train windows: {len(X_tr)}, Test windows: {len(X_te)}")

# =============================================
# 3. MODELS
# =============================================
class GraphConv(nn.Module):
    """Kipf & Welling: H' = sigma(A_hat @ H @ W)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    def forward(self, x, adj):
        return torch.matmul(adj, self.lin(x))

class TGCNModel(nn.Module):
    def __init__(self, n_nodes, hidden=32, win_out=3):
        super().__init__()
        self.n_nodes, self.hidden = n_nodes, hidden
        self.gconv = GraphConv(1, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, win_out)
    def forward(self, x, adj):
        B, T, N = x.shape
        h = torch.zeros(B*N, self.hidden, device=x.device)
        for t in range(T):
            xt = x[:, t, :].unsqueeze(-1)             # (B, N, 1)
            xt = F.relu(self.gconv(xt, adj))           # (B, N, hidden)
            h = self.gru(xt.reshape(B*N, self.hidden), h)
        out = self.out(h).reshape(B, N, -1).permute(0, 2, 1)
        return out                                     # (B, win_out, N)

class LSTMBaseline(nn.Module):
    def __init__(self, n_nodes, hidden=32, win_out=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, win_out)
    def forward(self, x, adj=None):
        B, T, N = x.shape
        x = x.permute(0, 2, 1).reshape(B*N, T, 1)
        out, _ = self.lstm(x)
        pred = self.out(out[:, -1, :]).reshape(B, N, -1).permute(0, 2, 1)
        return pred

# =============================================
# 4. TRAINING
# =============================================
device = torch.device("cpu")
print(f"\nDevice: {device}")
EPOCHS, BATCH = 20, 32

def train_model(model, name=""):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "test": []}
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        tr_loss = []
        for i in range(0, len(X_tr), BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X_tr[idx].to(device), Y_tr[idx].to(device)
            pred = model(xb, A_norm_t.to(device))
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss.append(loss.item())
        model.eval()
        te_loss = []
        with torch.no_grad():
            for i in range(0, len(X_te), BATCH):
                xb, yb = X_te[i:i+BATCH].to(device), Y_te[i:i+BATCH].to(device)
                pred = model(xb, A_norm_t.to(device))
                te_loss.append(loss_fn(pred, yb).item())
        tr, te = np.mean(tr_loss), np.mean(te_loss)
        hist["train"].append(tr); hist["test"].append(te)
        print(f"[{name}] Epoch {ep+1:02d} | train={tr:.5f} | test={te:.5f}")
    return model, hist

print("\n--- Training T-GCN ---")
tgcn, hist_tg = train_model(TGCNModel(NUM_NODES, 32, WIN_OUT), "TGCN")
print("\n--- Training LSTM baseline ---")
lstm, hist_ls = train_model(LSTMBaseline(NUM_NODES, 32, WIN_OUT), "LSTM")

# =============================================
# 5. EVALUATE
# =============================================
def collect(model):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH):
            xb = X_te[i:i+BATCH].to(device)
            preds.append(model(xb, A_norm_t.to(device)).cpu().numpy())
            truths.append(Y_te[i:i+BATCH].numpy())
    return np.concatenate(truths), np.concatenate(preds)

yt_tg, yp_tg = collect(tgcn)
yt_ls, yp_ls = collect(lstm)
def met(yt, yp):
    return np.sqrt(((yt-yp)**2).mean()), np.abs(yt-yp).mean()
rmse_tg, mae_tg = met(yt_tg, yp_tg)
rmse_ls, mae_ls = met(yt_ls, yp_ls)
print("\n=========== RESULTS ===========")
print(f"T-GCN  | RMSE = {rmse_tg:.5f}  MAE = {mae_tg:.5f}")
print(f"LSTM   | RMSE = {rmse_ls:.5f}  MAE = {mae_ls:.5f}")
print(f"T-GCN improves RMSE by {(1-rmse_tg/rmse_ls)*100:.1f}% over LSTM")
print("================================\n")

# =============================================
# 6. PLOTS
# =============================================
print("Generating figures...")
G = nx.from_numpy_array(A_raw - np.eye(NUM_NODES))

plt.figure(figsize=(9, 9))
pos = nx.spring_layout(G, seed=42, k=0.15)
nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5)
nx.draw_networkx_nodes(G, pos, node_size=30, node_color="#4361ee")
plt.title(f"METR-LA freeway sensor network — {NUM_NODES} sensors, {G.number_of_edges()} edges")
plt.axis("off"); plt.tight_layout()
plt.savefig("00_graph_topology.png", dpi=140); plt.close()

plt.figure(figsize=(8, 4))
plt.plot(hist_tg["test"], label="T-GCN", marker="o")
plt.plot(hist_ls["test"], label="LSTM", marker="s", linestyle="--")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Test loss curves")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("01_loss_curves.png", dpi=140); plt.close()

plt.figure(figsize=(12, 4))
plt.plot(yt_tg[:200, 0, 10], label="actual", alpha=0.85)
plt.plot(yp_tg[:200, 0, 10], label="T-GCN predicted", alpha=0.85)
plt.xlabel("Test snapshot (5-min)"); plt.ylabel("Speed (normalised)")
plt.title("Sensor 10 — 5-min ahead prediction")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("02_sensor_prediction.png", dpi=140); plt.close()

err = np.abs(yt_tg - yp_tg).mean(axis=(0, 1))
plt.figure(figsize=(12, 3))
plt.bar(range(len(err)), err, color="#4361ee", alpha=0.85)
plt.xlabel("Sensor"); plt.ylabel("Mean absolute error")
plt.title("Per-sensor prediction error (T-GCN)"); plt.tight_layout()
plt.savefig("03_per_sensor_error.png", dpi=140); plt.close()

plt.figure(figsize=(6, 4))
x = np.arange(2); w = 0.35
plt.bar(x-w/2, [rmse_tg, mae_tg], w, label="T-GCN", color="#4361ee")
plt.bar(x+w/2, [rmse_ls, mae_ls], w, label="LSTM", color="#f72585")
plt.xticks(x, ["RMSE", "MAE"]); plt.ylabel("Error")
plt.title("T-GCN vs LSTM (lower = better)")
plt.legend(); plt.tight_layout()
plt.savefig("04_model_comparison.png", dpi=140); plt.close()

# Bonus insight: degree vs error
degrees = (A_raw - np.eye(NUM_NODES)).sum(axis=1)
correlation = np.corrcoef(degrees, err)[0, 1]
print(f"Pearson correlation(node degree, error) = {correlation:.3f}")
plt.figure(figsize=(7, 5))
plt.scatter(degrees, err, s=40, color="#4361ee", alpha=0.6, edgecolor="black", linewidth=0.4)
plt.xlabel("Node degree (number of neighbours)")
plt.ylabel("Mean absolute prediction error")
plt.title(f"Node connectivity vs T-GCN error  (Pearson r = {correlation:.3f})")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("05_degree_vs_error.png", dpi=140); plt.close()

print("Saved 6 figures: 00_graph_topology.png, 01_loss_curves.png, 02_sensor_prediction.png, 03_per_sensor_error.png, 04_model_comparison.png, 05_degree_vs_error.png")
print("Done.")