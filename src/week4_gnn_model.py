"""
Week 4 — Train a Graph Attention model on the K-NN neighborhood graph
to predict house prices, then compare against the Week 2 XGBoost baseline.

This uses a lightweight, dependency-free attention mechanism built on
plain PyTorch tensors (no torch_geometric/DGL install required) — it
aggregates each house's neighbor features, weighting each neighbor by
a learned attention score, per the spec's "attention mechanism to weigh
certain neighbors more heavily than others."
"""

import pandas as pd
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

GRAPH_PATH = Path("models/house_graph.pkl")
NODE_FEATURES_PATH = Path("data/processed/graph_node_features.csv")
MODEL_PATH = Path("models/gat_model.pt")
METRICS_PATH = Path("outputs/week4_gnn_metrics.json")

TARGET_COL = "price"
FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "house_age", "was_renovated", "yrs_since_update",
    "dist_to_center_km", "neighbor_avg_price", "neighbor_price_std",
]

HIDDEN_DIM = 32
EPOCHS = 100
LR = 0.005


class GraphAttentionLayer(nn.Module):
    """Single-head attention: each node attends over its K neighbors,
    learning to weight some neighbors more than others."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, neighbor_idx):
        # x: [N, in_dim], neighbor_idx: [N, K]
        h = self.W(x)  # [N, out_dim]
        N, K = neighbor_idx.shape

        h_self = h.unsqueeze(1).expand(-1, K, -1)          # [N, K, out_dim]
        h_neighbors = h[neighbor_idx]                        # [N, K, out_dim]

        attn_input = torch.cat([h_self, h_neighbors], dim=-1)  # [N, K, 2*out_dim]
        e = self.leakyrelu(self.attn(attn_input)).squeeze(-1)  # [N, K]
        alpha = torch.softmax(e, dim=1)                          # attention weights

        out = torch.sum(alpha.unsqueeze(-1) * h_neighbors, dim=1)  # [N, out_dim]
        return out, alpha


class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, neighbor_idx):
        h, alpha1 = self.gat1(x, neighbor_idx)
        h = self.relu(h)
        h, alpha2 = self.gat2(h, neighbor_idx)
        h = self.relu(h)
        return self.out(h).squeeze(-1), alpha2


def build_neighbor_index(G, n_nodes, k):
    """Fixed-size [N, K] neighbor index matrix for tensor-friendly attention."""
    neighbor_idx = np.zeros((n_nodes, k), dtype=np.int64)
    for i in range(n_nodes):
        neighbors = list(G.neighbors(i))
        if len(neighbors) < k:
            neighbors = neighbors + [i] * (k - len(neighbors))  # pad with self
        neighbor_idx[i] = neighbors[:k]
    return neighbor_idx


def main():
    df = pd.read_csv(NODE_FEATURES_PATH).reset_index(drop=True)
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    K = 8  # must match Week 3's K_NEIGHBORS

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)
    y = df[TARGET_COL].values

    neighbor_idx = build_neighbor_index(G, len(df), K)

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    neighbor_idx_t = torch.tensor(neighbor_idx, dtype=torch.long)

    model = GATRegressor(in_dim=X_t.shape[1], hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True

    print(f"Training GAT on {len(train_idx):,} nodes, testing on {len(test_idx):,} nodes...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_t, neighbor_idx_t)
        loss = loss_fn(pred[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE Loss: {loss.item():,.2f}")

    model.eval()
    with torch.no_grad():
        pred, _ = model(X_t, neighbor_idx_t)
        y_pred_test = pred[test_idx].numpy()
        y_true_test = y_t[test_idx].numpy()

    mape = mean_absolute_percentage_error(y_true_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    print(f"\nGAT MAPE: {mape:.4f} ({mape * 100:.2f}%)")
    print(f"GAT RMSE: ${rmse:,.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    metrics = {
        "model": "Graph Attention Network",
        "mape": round(float(mape), 4),
        "rmse": round(float(rmse), 2),
        "n_test_samples": len(test_idx),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()