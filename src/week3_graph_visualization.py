"""
Week 3 — Visualize a small sample of the neighborhood graph to sanity-check
that edges connect geographically close houses.
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

GRAPH_PATH = Path("models/house_graph.pkl")
OUTPUT_PLOT_PATH = Path("outputs/week3_graph_sample.png")

SAMPLE_SIZE = 150  # keep small — full graph is unreadable when plotted


def main():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    sample_nodes = list(G.nodes())[:SAMPLE_SIZE]
    subG = G.subgraph(sample_nodes)

    pos = {n: (subG.nodes[n]["long"], subG.nodes[n]["lat"]) for n in subG.nodes()}

    plt.figure(figsize=(10, 8))
    nx.draw(
        subG, pos, node_size=25, node_color="steelblue",
        edge_color="lightgray", width=0.6, with_labels=False,
    )
    plt.title(f"Sample Neighborhood Graph ({SAMPLE_SIZE} houses, K-NN edges)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved graph visualization -> {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()