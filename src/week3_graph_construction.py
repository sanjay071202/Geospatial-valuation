"""
Week 3 — Build a K-nearest-neighbor graph connecting each house to its
closest physical neighbors, using Haversine distance as edge weight.
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from sklearn.neighbors import BallTree

FEATURED_DATA_PATH = Path("data/processed/kc_house_features.csv")
GRAPH_PATH = Path("models/house_graph.pkl")
NODE_FEATURES_PATH = Path("data/processed/graph_node_features.csv")

K_NEIGHBORS = 8  # number of nearest neighbors per house


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_knn_graph(df: pd.DataFrame, k: int) -> nx.Graph:
    coords = np.radians(df[["lat", "long"]].values)

    # BallTree with haversine metric — efficient KNN search for lat/long
    tree = BallTree(coords, metric="haversine")
    distances, indices = tree.query(coords, k=k + 1)  # +1 because point itself is included

    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, price=row["price"], lat=row["lat"], long=row["long"])

    for i in range(len(df)):
        for j_pos in range(1, k + 1):  # skip index 0 (self)
            neighbor_idx = indices[i][j_pos]
            dist_km = distances[i][j_pos] * 6371.0  # convert radians back to km
            G.add_edge(i, neighbor_idx, weight=dist_km)

    return G


def compute_neighbor_price_features(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """For each house, compute summary stats of its K nearest neighbors' prices —
    a simple spatial embedding proxy before the full GNN in Week 4."""
    neighbor_avg_price = []
    neighbor_price_std = []

    for idx in df.index:
        neighbors = list(G.neighbors(idx))
        neighbor_prices = df.loc[neighbors, "price"]
        neighbor_avg_price.append(neighbor_prices.mean())
        neighbor_price_std.append(neighbor_prices.std())

    df = df.copy()
    df["neighbor_avg_price"] = neighbor_avg_price
    df["neighbor_price_std"] = neighbor_price_std
    return df


def main():
    df = pd.read_csv(FEATURED_DATA_PATH).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows for graph construction")

    G = build_knn_graph(df, k=K_NEIGHBORS)
    print(f"Built graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"(K={K_NEIGHBORS} neighbors per house)")

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph -> {GRAPH_PATH}")

    df_with_embeddings = compute_neighbor_price_features(df, G)
    NODE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_with_embeddings.to_csv(NODE_FEATURES_PATH, index=False)
    print(f"Saved node features with spatial context -> {NODE_FEATURES_PATH}")


if __name__ == "__main__":
    main()