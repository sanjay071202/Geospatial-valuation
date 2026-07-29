"""
Week 4 — Streamlit dashboard visualizing GAT price predictions vs actual
prices on an interactive map.

Run with: streamlit run src/week4_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import pydeck as pdk
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from week4_gnn_model import GATRegressor, build_neighbor_index, FEATURE_COLS, TARGET_COL

NODE_FEATURES_PATH = Path("data/processed/graph_node_features.csv")
GRAPH_PATH = Path("models/house_graph.pkl")
MODEL_PATH = Path("models/gat_model.pt")
K = 8
HIDDEN_DIM = 32

st.set_page_config(page_title="Geospatial Valuation Dashboard", layout="wide")
st.title("Geospatial Real Estate Valuation — Prediction Disparity Map")


@st.cache_data
def load_data():
    df = pd.read_csv(NODE_FEATURES_PATH).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return df, G


@st.cache_resource
def load_model(in_dim):
    model = GATRegressor(in_dim=in_dim, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


df, G = load_data()
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURE_COLS].values)
X_t = torch.tensor(X, dtype=torch.float32)
neighbor_idx_t = torch.tensor(build_neighbor_index(G, len(df), K), dtype=torch.long)

model = load_model(in_dim=X_t.shape[1])
with torch.no_grad():
    pred, _ = model(X_t, neighbor_idx_t)
    df["predicted_price"] = pred.numpy()

df["price_diff_pct"] = ((df["predicted_price"] - df["price"]) / df["price"]) * 100

col1, col2 = st.columns(2)
col1.metric("Avg Actual Price", f"${df['price'].mean():,.0f}")
col2.metric("Avg Predicted Price", f"${df['predicted_price'].mean():,.0f}")

st.subheader("Predicted vs Actual Price Disparity Map")
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=df["lat"].mean(), longitude=df["long"].mean(), zoom=9,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[long, lat]",
                get_color="[255, (1 - abs(price_diff_pct)/50) * 255, 0, 160]",
                get_radius=80,
            ),
        ],
    )
)

st.subheader("Sample Predictions")
st.dataframe(
    df[["price", "predicted_price", "price_diff_pct"]].sample(20).round(2)
)