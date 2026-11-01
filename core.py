"""
AI-Based Cybersecurity Monitoring & Attack Path Prediction  (v3.1) - core logic
================================================================
Part 1  (Machine Learning)  : Random Forest classifies network flows as
                              BENIGN / MALICIOUS, properly evaluated on data
                              the model has never seen (train/test split).
Part 2  (Search / AI agent) : BFS, DFS and A* look for an attack path through
                              a simulated corporate network.
LINK    (ML -> Search)      : the ML model's predicted "malicious probability"
                              of observed traffic sets the threat level, which
                              changes the risk (and cost) of the exposed nodes
                              the search algorithms use.

This module has NO Streamlit code, so it can be tested with pytest and reused
by experiments.py. The user interface is in app.py.
"""
import io
import heapq
import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MAX_ROWS = 60_000        # rows kept for training (keeps the app fast)
PER_CHUNK = 8_000        # rows sampled from each 50k-row chunk of the CSV
CHUNK_SIZE = 50_000      # rows per chunk when reading the CSV
MIN_EDGE_COST = 0.05     # no node is ever "free" to enter (needed for A*)
PORT_FEATURES = ("destination port", "dst port")   # used by the optional port-ablation experiment

# --------------------------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# --------------------------------------------------------------------------


def _sample_chunks(source):
    """Read a (possibly huge) CSV in chunks and keep a RANDOM sample of every chunk.

    Why: capture files are ordered (CIC-IDS2017 daily files follow the clock, so
    attacks cluster in time and the first rows of a file can all be benign).
    Reading only the first N rows could give a model that has never seen an
    attack. Sampling from every chunk avoids that. Each chunk contributes the same
    number of rows, so class shares in the sample can differ from the full file.
    """
    reader = pd.read_csv(source, chunksize=CHUNK_SIZE, low_memory=False)
    parts = []
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        parts.append(chunk.sample(n=min(PER_CHUNK, len(chunk)), random_state=RANDOM_STATE))
    df = pd.concat(parts, ignore_index=True)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def make_demo_data(n=6000, seed=RANDOM_STATE):
    """SYNTHETIC data with CICIDS-like column names, only to test the app.
    Never report accuracy obtained on this data in your project."""
    rng = np.random.default_rng(seed)
    n_att = n // 2
    n_ben = n - n_att
    ben = pd.DataFrame({
        "Dst Port": rng.choice([80, 443, 53, 22, 8080], n_ben),
        "Flow Duration": rng.exponential(2e6, n_ben),
        "Tot Fwd Pkts": rng.poisson(12, n_ben),
        "Tot Bwd Pkts": rng.poisson(10, n_ben),
        "Fwd Pkt Len Max": rng.normal(500, 250, n_ben).clip(0),
        "Flow IAT Mean": rng.exponential(2e5, n_ben),
        "ACK Flag Cnt": rng.binomial(1, 0.6, n_ben),
        "Init Fwd Win Byts": rng.normal(8000, 4000, n_ben).clip(0),
    })
    att = pd.DataFrame({
        "Dst Port": rng.choice([80, 443, 8080], n_att),
        "Flow Duration": rng.exponential(6e5, n_att),
        "Tot Fwd Pkts": rng.poisson(6, n_att),
        "Tot Bwd Pkts": rng.poisson(3, n_att),
        "Fwd Pkt Len Max": rng.normal(300, 200, n_att).clip(0),
        "Flow IAT Mean": rng.exponential(8e4, n_att),
        "ACK Flag Cnt": rng.binomial(1, 0.3, n_att),
        "Init Fwd Win Byts": rng.normal(4000, 3500, n_att).clip(0),
    })
    ben["Label"] = "Benign"
    att["Label"] = "DoS attacks-Hulk"
    return pd.concat([ben, att], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare(df):
    """Turn a raw CICIDS-style DataFrame into (X, y, label_names).

    y = 0 for benign traffic, 1 for any attack.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    label_col = "Label" if "Label" in df.columns else df.columns[-1]

    # Some CICIDS exports (for example the 2018 files) repeat the header row inside the data.
    df = df[df[label_col].astype(str).str.strip() != label_col]

    label_text = df[label_col].astype(str).str.strip()
    y = (~label_text.str.upper().str.contains("BENIGN")).astype(int)

    drop = [label_col] + [c for c in df.columns
                          if c.lower().startswith("unnamed") or c.lower() in
                          ("timestamp", "flow id", "source ip", "destination ip", "src ip", "dst ip")]
    X = df.drop(columns=drop, errors="ignore").apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all").fillna(0)   # unreadable values -> 0 (simple, documented choice)
    return X.reset_index(drop=True), y.reset_index(drop=True), label_text.reset_index(drop=True)


def build_pipeline(_source, drop_features=()):
    """Load -> prepare -> split -> train -> evaluate.

    drop_features: optional column names (case-insensitive) removed before training,
    used by the ablation experiment (for example PORT_FEATURES). Default: use all features."""
    if isinstance(_source, pd.DataFrame):
        df = _source
    elif _source == "demo":
        df = make_demo_data()
    else:
        df = _sample_chunks(io.BytesIO(_source) if isinstance(_source, bytes) else _source)
    X, y, label_text = prepare(df)
    if drop_features:
        drop = {d.lower() for d in drop_features}
        X = X.drop(columns=[c for c in X.columns if c.lower() in drop])
    if y.nunique() < 2:
        raise ValueError(
            "The sampled data contains only ONE class "
            f"({'benign' if y.iloc[0] == 0 else 'attack'}). "
            "Use a file with both benign and attack rows.")

    # 70 % of the flows teach the model, 30 % are kept hidden to test it.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                   class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    metrics = dict(accuracy=accuracy_score(y_te, pred), precision=prec, recall=rec, f1=f1,
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   n_train=len(X_tr), n_test=len(X_te))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(model=model, X_test=X_te.reset_index(drop=True), y_test=y_te.reset_index(drop=True),
                proba_test=proba, metrics=metrics, importances=importances,
                class_counts=label_text.value_counts(), n_features=X.shape[1])


# --------------------------------------------------------------------------
# 2. THE SIMULATED NETWORK AND THE SEARCH ALGORITHMS
# --------------------------------------------------------------------------

EDGES = [
    ("Internet", "Edge_Router"), ("Edge_Router", "WAF"),
    ("WAF", "DMZ_Web"), ("WAF", "DMZ_App"),
    ("DMZ_Web", "Core_Switch"), ("DMZ_App", "Core_Switch"),
    ("Core_Switch", "HR_Workstation"), ("Core_Switch", "Dev_Workstation"),
    ("HR_Workstation", "Internal_DB"), ("Dev_Workstation", "Internal_DB"),
    ("HR_Workstation", "Backup_Server"), ("Dev_Workstation", "Backup_Server"),
]
# ASSUMPTION (hand-set, to be justified in your report): base vulnerability of each
# node, 0 = very hard to compromise, 1 = very easy.
BASE_VULN = {"Internet": 0.0, "Edge_Router": 0.10, "WAF": 0.05, "DMZ_Web": 0.80, "DMZ_App": 0.60,
             "Core_Switch": 0.10, "HR_Workstation": 0.90, "Dev_Workstation": 0.20,
             "Internal_DB": 0.70, "Backup_Server": 0.40}
# Nodes reachable from the Internet: the observed attack traffic raises THEIR risk.
EXPOSED = {"Edge_Router", "WAF", "DMZ_Web", "DMZ_App"}
LAYER = {"Internet": 0, "Edge_Router": 1, "WAF": 2, "DMZ_Web": 3, "DMZ_App": 3, "Core_Switch": 4,
         "HR_Workstation": 5, "Dev_Workstation": 5, "Internal_DB": 6, "Backup_Server": 6}


def build_graph():
    G = nx.DiGraph()
    G.add_edges_from(EDGES)
    return G


def node_risk(threat):
    """Risk of each node in [0, 0.95].
    Exposed nodes: base * (0.5 + threat)  -> threat 0 halves, threat 1 x1.5.
    Internal nodes: base vulnerability only."""
    risk = {}
    for n, v in BASE_VULN.items():
        r = v * (0.5 + threat) if n in EXPOSED else v
        risk[n] = min(0.95, r)
    return risk


def node_cost(risk):
    """Cost for the ATTACKER of entering a node: easy-to-hack node = cheap."""
    return {n: max(MIN_EDGE_COST, 1.0 - r) for n, r in risk.items()}


def path_cost(path, cost):
    return sum(cost[n] for n in path[1:])


def bfs_search(G, start, goal):
    """Breadth-first: fewest hops, ignores risk."""
    queue, parent, expanded = [start], {start: None}, 0
    while queue:
        node = queue.pop(0)
        expanded += 1
        if node == goal:
            return _rebuild(parent, goal), expanded
        for nb in G.neighbors(node):
            if nb not in parent:
                parent[nb] = node
                queue.append(nb)
    return None, expanded


def dfs_search(G, start, goal):
    """Depth-first: goes deep first, ignores risk, not guaranteed shortest."""
    stack, visited, expanded = [[start]], set(), 0
    while stack:
        path = stack.pop()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
        if node == goal:
            return path, expanded
        for nb in G.neighbors(node):
            if nb not in visited:
                stack.append(path + [nb])
    return None, expanded


def astar_search(G, start, goal, cost):
    """A*: f(n) = g(n) + h(n).
    g(n) = attacker cost paid so far.
    h(n) = (hops still needed to reach the goal) x MIN_EDGE_COST. It never
    over-estimates the true remaining cost (every node costs >= MIN_EDGE_COST),
    so A* is guaranteed to return the CHEAPEST path (optimal)."""
    hops = nx.single_source_shortest_path_length(G.reverse(copy=False), goal)
    h = {n: d * MIN_EDGE_COST for n, d in hops.items()}
    heap = [(h.get(start, 0.0), 0.0, start)]
    g, parent, closed, expanded = {start: 0.0}, {start: None}, set(), 0
    while heap:
        _, gv, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            return _rebuild(parent, goal), expanded
        for nb in G.neighbors(node):
            ng = gv + cost[nb]
            if ng < g.get(nb, math.inf):
                g[nb], parent[nb] = ng, node
                heapq.heappush(heap, (ng + h.get(nb, 0.0), ng, nb))
    return None, expanded


def _rebuild(parent, goal):
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return path[::-1]


def mean_runtime_us(fn, repeats=300):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1e6


def draw_network(G, risk, path, title, figsize=(11, 4.8), font_size=7, node_size=2300):
    pos = {}
    by_layer = {}
    for n, l in LAYER.items():
        by_layer.setdefault(l, []).append(n)
    for l, nodes in by_layer.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (l * 1.7, (len(nodes) - 1) / 2 - i)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [plt.cm.YlOrRd(risk[n]) for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_size, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, node_size=node_size, ax=ax)
    for n in G.nodes:   # white text on dark nodes so it stays readable
        nx.draw_networkx_labels(G, pos, labels={n: n.replace("_", "\n")}, font_size=font_size, ax=ax,
                                font_color="white" if risk[n] > 0.6 else "black")
    if path and len(path) > 1:
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", width=3.5,
                               arrows=True, node_size=node_size, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="none", edgecolors="red",
                               linewidths=3, node_size=node_size, ax=ax)
    ax.set_title(title + "   (node colour = risk: yellow low -> dark red high)")
    ax.margins(0.12)
    ax.axis("off")
    return fig


