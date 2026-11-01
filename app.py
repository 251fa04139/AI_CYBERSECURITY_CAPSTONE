"""
Streamlit user interface (v3.1). All logic lives in core.py.
Run:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

from core import (RANDOM_STATE, build_pipeline as _build_pipeline, build_graph, node_risk, node_cost,
                  path_cost, bfs_search, dfs_search, astar_search, mean_runtime_us, draw_network)


@st.cache_resource(show_spinner="Reading data and training the Random Forest ...")
def build_pipeline(source_key, _source):
    """Cached per data source (source_key); _source is not hashed."""
    return _build_pipeline(_source)



st.set_page_config(page_title="AI Cybersecurity Agent", layout="wide")
st.title("🛡️ AI-Based Cybersecurity Monitoring & Attack Path Prediction")

st.sidebar.header("1. Data source")
uploaded = st.sidebar.file_uploader("Upload a CIC-IDS2017-style CSV (e.g. Tuesday-WorkingHours.pcap_ISCX.csv)", type=["csv"])
path_text = st.sidebar.text_input("...or full path to a CSV on this computer (for very large files)")
use_demo = st.sidebar.checkbox("Use built-in SYNTHETIC demo data (testing only)")

if uploaded is not None:
    source_key, source = f"upload:{uploaded.name}:{uploaded.size}", uploaded.getvalue()
elif path_text.strip():
    source_key, source = f"path:{path_text.strip()}", path_text.strip()
elif use_demo:
    source_key, source = "demo", "demo"
else:
    st.info("👈 Upload a CIC-IDS2017 CSV (for example the Tuesday, Wednesday or Friday-DDos file; Monday is all benign and will not work), "
            "give a file path, or tick the synthetic demo box.")
    st.stop()

try:
    P = build_pipeline(source_key, source)
except Exception as e:  # show a readable message instead of a crash
    st.error(f"Could not process the data: {e}")
    st.stop()

if source_key == "demo":
    st.warning("Synthetic demo data: use it only to test the app. Do NOT report these numbers.")

# ---- Section 1: model evaluation ----------------------------------------
m = P["metrics"]
st.header("1. Machine Learning: Random Forest evaluation")
st.caption(f"{m['n_train']:,} flows used to TRAIN the model, {m['n_test']:,} unseen flows used to TEST it. "
           f"{P['n_features']} numeric features. Labels: BENIGN = 0, any attack = 1.")
c = st.columns(4)
c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
c[1].metric("Precision", f"{m['precision']:.3f}")
c[2].metric("Recall (attacks caught)", f"{m['recall']:.3f}")
c[3].metric("F1-score", f"{m['f1']:.3f}")

left, right = st.columns(2)
with left:
    st.write("**Confusion matrix (test set)**")
    st.table(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                          index=["Actually BENIGN", "Actually ATTACK"],
                          columns=["Predicted BENIGN", "Predicted ATTACK"]))
    st.caption(f"False alarms (FP) = {m['fp']}; missed attacks (FN) = {m['fn']}. "
               "In security, a missed attack is usually worse than a false alarm.")
    st.write("**Class distribution in the sample**")
    st.dataframe(P["class_counts"].rename("flows"))
with right:
    st.write("**Top 10 most important features**")
    st.bar_chart(P["importances"].head(10))

# ---- Section 2: single-flow inspector -----------------------------------
st.header("2. Real-time threat detection on one flow")
idx = st.slider("Select an unseen test flow", 0, len(P["X_test"]) - 1, 0)
row = P["X_test"].iloc[[idx]]
p_mal = float(P["model"].predict_proba(row)[0, 1])
truth = "ATTACK" if P["y_test"].iloc[idx] == 1 else "BENIGN"
a, b, d = st.columns(3)
a.metric("Model verdict", "🚨 MALICIOUS" if p_mal >= 0.5 else "✅ BENIGN")
b.metric("P(malicious)", f"{p_mal:.2%}")
d.metric("True label", truth, delta="correct" if (p_mal >= 0.5) == (truth == "ATTACK") else "WRONG",
         delta_color="normal" if (p_mal >= 0.5) == (truth == "ATTACK") else "inverse")

# ---- Section 3: ML -> threat level -> search ----------------------------
st.header("3. AI agent: attack-path search driven by the ML model")
scenario = st.radio("Observed traffic window (200 flows from the test set)",
                    ("Mixed traffic", "Only benign flows", "Only attack flows"), horizontal=True)
rng = np.random.default_rng(RANDOM_STATE)
if scenario == "Only benign flows":
    pool = np.where(P["y_test"].values == 0)[0]
elif scenario == "Only attack flows":
    pool = np.where(P["y_test"].values == 1)[0]
else:
    pool = np.arange(len(P["y_test"]))
sel = rng.choice(pool, size=min(200, len(pool)), replace=False)
threat = float(P["proba_test"][sel].mean())
st.metric("Threat level (mean P(malicious) predicted by the Random Forest on this window)", f"{threat:.2f}")

G = build_graph()
risk = node_risk(threat)
cost = node_cost(risk)
nodes = list(G.nodes)
col_s, col_g = st.columns(2)
start = col_s.selectbox("Attacker starts at", nodes, index=nodes.index("Internet"))
goal = col_g.selectbox("Attacker target", nodes, index=nodes.index("Internal_DB"))

if start == goal or not nx.has_path(G, start, goal):
    st.error("Choose a start and a target such that a path exists from start to target.")
    st.stop()

p_bfs, e_bfs = bfs_search(G, start, goal)
p_dfs, e_dfs = dfs_search(G, start, goal)
p_ast, e_ast = astar_search(G, start, goal, cost)
reference = nx.astar_path(G, start, goal, weight=lambda u, v, dd: cost[v])
assert abs(path_cost(reference, cost) - path_cost(p_ast, cost)) < 1e-9, "A* is not optimal!"

results = [("BFS (uninformed)", p_bfs, e_bfs, lambda: bfs_search(G, start, goal)),
           ("DFS (uninformed)", p_dfs, e_dfs, lambda: dfs_search(G, start, goal)),
           ("A* (risk-informed)", p_ast, e_ast, lambda: astar_search(G, start, goal, cost))]
table = pd.DataFrame({
    "Algorithm": [r[0] for r in results],
    "Path found": [" ➔ ".join(r[1]) for r in results],
    "Hops": [len(r[1]) - 1 for r in results],
    "Attacker cost (lower = easier attack)": [round(path_cost(r[1], cost), 3) for r in results],
    "Nodes expanded": [r[2] for r in results],
    "Mean time (µs)": [round(mean_runtime_us(r[3]), 1) for r in results],
})
st.write("**📊 Algorithm comparison**")
st.table(table)
st.caption("A* cost matches NetworkX's reference result (verified in the code). "
           "BFS and DFS ignore risk: they may find a valid path that is more expensive for the attacker. "
           "On a 10-node graph all times are tiny; the meaningful metrics are path cost and nodes expanded.")

choice = st.radio("Visualize path of:", [r[0] for r in results], horizontal=True)
chosen = next(r[1] for r in results if r[0] == choice)
st.pyplot(draw_network(G, risk, chosen, f"{choice}: {' ➔ '.join(chosen)}"))

with st.expander("How is the ML model linked to the search? (read before presenting)"):
    st.markdown(f"""
1. The Random Forest gives each observed flow a probability of being malicious.
2. The **threat level** is the average of these probabilities over the selected traffic window (now **{threat:.2f}**).
3. For nodes exposed to the Internet (`Edge_Router, WAF, DMZ_Web, DMZ_App`): `risk = base_vulnerability × (0.5 + threat)`.
   Internal nodes keep their base vulnerability (**hand-assigned assumption** in `BASE_VULN`).
4. `cost(node) = max(0.05, 1 − risk)`: the easier a node is to compromise, the cheaper it is for the attacker.
5. A\\* minimises the total cost with `f(n) = g(n) + h(n)`, where `h(n) = remaining hops × 0.05` never over-estimates, so the result is optimal.

**Limitations:** the network is simulated, base vulnerabilities are assumptions, and one global threat level is applied
to all exposed nodes. The system predicts the *most likely cheapest path*, not the attacker's actual behaviour.
""")
