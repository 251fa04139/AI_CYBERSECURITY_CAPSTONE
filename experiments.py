"""
Reproducible experiments.  Run from the project folder.

  python experiments.py search                 # no dataset needed
  python experiments.py ml <file.csv>          # REAL data: compares 3 models, writes results/ml_results.csv
  python experiments.py ml --demo              # SYNTHETIC data, only to test that the script works
  python experiments.py real <f1.csv> <f2.csv> ...   # REAL metrics -> results/real_metrics.json (used by apply_results.py)
  python experiments.py ablate <f1.csv> ...          # same, with and without the Destination Port feature -> results/ablation_metrics.json

Outputs go to results/. Numbers from --demo must NEVER be reported.
"""
import sys, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import core

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def search_benchmark():
    """E1: attacker cost / nodes expanded of BFS, DFS, A* on the 10-node network, threat 0 -> 1.
    E2: same algorithms on random layered graphs of growing size (30 graphs each)."""
    G = core.build_graph()
    rows = []
    for threat in np.round(np.linspace(0, 1, 11), 2):
        cost = core.node_cost(core.node_risk(threat))
        for name, fn in (("BFS", lambda: core.bfs_search(G, "Internet", "Internal_DB")),
                         ("DFS", lambda: core.dfs_search(G, "Internet", "Internal_DB")),
                         ("A*", lambda: core.astar_search(G, "Internet", "Internal_DB", cost))):
            p, e = fn()
            rows.append(dict(threat=threat, algorithm=name, cost=round(core.path_cost(p, cost), 4),
                             hops=len(p) - 1, expanded=e, path=" > ".join(p)))
    e1 = pd.DataFrame(rows)
    e1.to_csv(f"{OUT}/e1_threat_sweep.csv", index=False)

    rows = []
    for layers, width in ((5, 3), (8, 5), (12, 8), (16, 12)):
        for seed in range(30):
            rng = random.Random(seed)
            H = nx.DiGraph()
            lay = [[f"L{l}_{i}" for i in range(width)] for l in range(layers)]
            for l in range(layers - 1):
                for u in lay[l]:
                    for v in rng.sample(lay[l + 1], k=min(width, 2)):
                        H.add_edge(u, v)
            src, dst = "S", "T"
            for u in lay[0]:
                H.add_edge(src, u)
            for u in lay[-1]:
                H.add_edge(u, dst)
            cost = {n: rng.uniform(0.05, 1.0) for n in H.nodes}
            if not nx.has_path(H, src, dst):
                continue
            ref = core.path_cost(nx.dijkstra_path(H, src, dst, weight=lambda a, b, d: cost[b]), cost)
            for name, fn in (("BFS", lambda: core.bfs_search(H, src, dst)),
                             ("DFS", lambda: core.dfs_search(H, src, dst)),
                             ("A*", lambda: core.astar_search(H, src, dst, cost))):
                p, e = fn()
                rows.append(dict(nodes=H.number_of_nodes(), algorithm=name, expanded=e,
                                 cost_ratio=core.path_cost(p, cost) / ref))
    e2 = pd.DataFrame(rows)
    summ = e2.groupby(["nodes", "algorithm"]).agg(expanded_mean=("expanded", "mean"),
                                                  cost_ratio_mean=("cost_ratio", "mean"),
                                                  cost_ratio_max=("cost_ratio", "max"),
                                                  graphs=("expanded", "size")).round(3).reset_index()
    summ.to_csv(f"{OUT}/e2_random_graphs.csv", index=False)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, g in e1.groupby("algorithm"):
        ax[0].plot(g["threat"], g["cost"], marker="o", label=name)
    ax[0].set_xlabel("threat level (from ML)"); ax[0].set_ylabel("attacker cost (lower = easier)")
    ax[0].set_title("E1: 10-node network"); ax[0].legend()
    for name, g in summ.groupby("algorithm"):
        ax[1].plot(g["nodes"], g["expanded_mean"], marker="o", label=name)
    ax[1].set_xlabel("graph size (nodes)"); ax[1].set_ylabel("mean nodes expanded"); ax[1].set_title("E2: random layered graphs"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/search_benchmark.png", dpi=130)
    print(e1.to_string(index=False)); print(); print(summ.to_string(index=False))


def ml_experiment(source):
    """Compare Decision Tree, Logistic Regression and Random Forest on the SAME split."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    import time
    df = core.make_demo_data() if source == "--demo" else core._sample_chunks(source)
    X, y, _ = core.prepare(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=core.RANDOM_STATE)
    models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
              "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=core.RANDOM_STATE),
              "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, class_weight="balanced", random_state=core.RANDOM_STATE)}
    rows = []
    for name, m in models.items():
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); t_fit = time.perf_counter() - t0
        pred = m.predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
        rows.append(dict(model=name, accuracy=accuracy_score(y_te, pred), precision=p, recall=r, f1=f,
                         train_seconds=t_fit, n_train=len(X_tr), n_test=len(X_te),
                         data="SYNTHETIC - DO NOT REPORT" if source == "--demo" else os.path.basename(source)))
    out = pd.DataFrame(rows).round(4)
    fn = "ml_results_DEMO.csv" if source == "--demo" else "ml_results.csv"
    out.to_csv(f"{OUT}/{fn}", index=False)
    print(out.to_string(index=False))


def ablation(paths):
    """Same pipeline with and without the Destination Port feature (tests whether the model just learned a port)."""
    import json, datetime
    out = {"_meta": {"date": datetime.date.today().isoformat(), "dropped": list(core.PORT_FEATURES), "seed": core.RANDOM_STATE}}
    rows = []
    for p in paths:
        keep = ("accuracy", "precision", "recall", "f1", "fp", "fn", "n_test")
        a = core.build_pipeline(p)["metrics"]
        b = core.build_pipeline(p, drop_features=core.PORT_FEATURES)["metrics"]
        out[os.path.basename(p)] = {"with_port": {k: a[k] for k in keep}, "without_port": {k: b[k] for k in keep}}
        for name, m in (("with port", a), ("without port", b)):
            rows.append(dict(file=os.path.basename(p)[:30], setting=name, accuracy=round(m["accuracy"], 4),
                             recall=round(m["recall"], 4), f1=round(m["f1"], 4), missed_FN=m["fn"], false_alarms_FP=m["fp"]))
    json.dump(out, open(f"{OUT}/ablation_metrics.json", "w"), indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved results/ablation_metrics.json  ->  now run: python apply_results.py")


def real_metrics(paths):
    """Run the app's own pipeline (core.build_pipeline) on each CSV and save the metrics."""
    import json, datetime
    out = {"_meta": {"date": datetime.date.today().isoformat(), "max_rows": core.MAX_ROWS, "split": "70/30 stratified",
                     "seed": core.RANDOM_STATE}}
    for p in paths:
        P = core.build_pipeline(p)
        m = P["metrics"]
        out[os.path.basename(p)] = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in m.items()}
        out[os.path.basename(p)]["n_features"] = int(P["n_features"])
    json.dump(out, open(f"{OUT}/real_metrics.json", "w"), indent=2)
    print(pd.DataFrame({k: v for k, v in out.items() if k != "_meta"}).T.round(4).to_string())
    print("saved results/real_metrics.json  ->  now run: python apply_results.py")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "search":
        search_benchmark()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ablate":
        ablation(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "real":
        real_metrics(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ml":
        ml_experiment(sys.argv[2])
    else:
        print(__doc__)
