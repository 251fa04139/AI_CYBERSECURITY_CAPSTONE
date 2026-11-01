"""Unit tests for core.py (standard library unittest, nothing extra to install).
Run from the project folder:   python -m unittest discover -s tests -v
"""
import os, sys, tempfile, unittest
import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core


class TestData(unittest.TestCase):
    def test_demo_data_has_two_classes(self):
        X, y, _ = core.prepare(core.make_demo_data())
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(X.shape[1], 8)

    def test_prepare_cleans_inf_nan_header_rows_and_ids(self):
        df = pd.DataFrame({"Flow ID": ["a", "b", "c", "d"], "Timestamp": ["t"] * 4,
                           "Dst Port": [80, "Dst Port", 443, 22],
                           "Flow Duration": [1.0, 2.0, np.inf, "x"],
                           "Label": ["Benign", "Label", "DDOS attack-HOIC", "Benign"]})
        X, y, _ = core.prepare(df)
        self.assertEqual(len(X), 3)                       # repeated header row removed
        self.assertNotIn("Flow ID", X.columns)
        self.assertNotIn("Timestamp", X.columns)
        self.assertTrue(np.isfinite(X.values).all())
        self.assertEqual(list(y), [0, 1, 0])

    def test_sorted_csv_still_gives_both_classes(self):
        df = core.make_demo_data(n=3000).sort_values("Label")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sorted.csv")
            df.to_csv(p, index=False)
            old = (core.PER_CHUNK, core.CHUNK_SIZE)
            try:
                core.CHUNK_SIZE, core.PER_CHUNK = 500, 200   # 6 chunks; first 3 are one class only
                sample = core._sample_chunks(p)
            finally:
                core.PER_CHUNK, core.CHUNK_SIZE = old
        self.assertEqual(sample["Label"].nunique(), 2)
        self.assertEqual(len(sample), 1200)

    def test_single_class_raises(self):
        df = core.make_demo_data(n=400)
        df = df[df["Label"] == "Benign"]
        with self.assertRaisesRegex(ValueError, "ONE class"):
            core.build_pipeline(df)


class TestModel(unittest.TestCase):
    def test_pipeline_metrics_are_consistent(self):
        r = core.build_pipeline("demo")
        m = r["metrics"]
        self.assertEqual(m["n_train"] + m["n_test"], 6000)
        self.assertEqual(m["tn"] + m["fp"] + m["fn"] + m["tp"], m["n_test"])
        self.assertAlmostEqual(m["accuracy"], (m["tp"] + m["tn"]) / m["n_test"])
        self.assertEqual(len(r["proba_test"]), m["n_test"])
        self.assertEqual(len(r["X_test"]), m["n_test"])

    def test_drop_features_removes_the_port_column(self):
        full = core.build_pipeline("demo")
        no_port = core.build_pipeline("demo", drop_features=core.PORT_FEATURES)
        self.assertIn("Dst Port", full["importances"].index)
        self.assertNotIn("Dst Port", no_port["importances"].index)
        self.assertEqual(no_port["n_features"], full["n_features"] - 1)
        self.assertEqual(no_port["metrics"]["n_test"], full["metrics"]["n_test"])

    def test_deterministic_with_fixed_seed(self):
        a = core.build_pipeline("demo")["metrics"]
        b = core.build_pipeline("demo")["metrics"]
        self.assertEqual(a, b)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = core.build_graph()

    def test_risk_and_cost_bounds(self):
        for t in (0.0, 0.5, 1.0):
            risk = core.node_risk(t)
            cost = core.node_cost(risk)
            self.assertTrue(all(0 <= r <= 0.95 for r in risk.values()))
            self.assertTrue(all(c >= core.MIN_EDGE_COST for c in cost.values()))

    def test_higher_threat_lowers_exposed_cost_only(self):
        lo = core.node_cost(core.node_risk(0.0))
        hi = core.node_cost(core.node_risk(1.0))
        self.assertLess(hi["DMZ_Web"], lo["DMZ_Web"])
        self.assertEqual(hi["Internal_DB"], lo["Internal_DB"])

    def test_astar_is_optimal_for_every_pair(self):
        G = self.G
        for threat in (0.0, 0.3, 0.7, 1.0):
            cost = core.node_cost(core.node_risk(threat))
            for a in G.nodes:
                for b in G.nodes:
                    if a != b and nx.has_path(G, a, b):
                        with self.subTest(threat=threat, a=a, b=b):
                            p, _ = core.astar_search(G, a, b, cost)
                            ref = nx.dijkstra_path(G, a, b, weight=lambda u, v, d: cost[v])
                            self.assertAlmostEqual(core.path_cost(p, cost), core.path_cost(ref, cost))

    def test_bfs_dfs_valid_and_bfs_fewest_hops(self):
        G = self.G
        pb, _ = core.bfs_search(G, "Internet", "Internal_DB")
        pd_, _ = core.dfs_search(G, "Internet", "Internal_DB")
        for p in (pb, pd_):
            self.assertEqual((p[0], p[-1]), ("Internet", "Internal_DB"))
            self.assertTrue(all(G.has_edge(u, v) for u, v in zip(p[:-1], p[1:])))
        self.assertEqual(len(pb) - 1, nx.shortest_path_length(G, "Internet", "Internal_DB"))

    def test_unreachable_returns_none(self):
        G = self.G
        cost = core.node_cost(core.node_risk(0.5))
        self.assertIsNone(core.bfs_search(G, "Internal_DB", "Internet")[0])
        self.assertIsNone(core.dfs_search(G, "Internal_DB", "Internet")[0])
        self.assertIsNone(core.astar_search(G, "Internal_DB", "Internet", cost)[0])

    def test_dfs_never_cheaper_than_astar(self):
        cost = core.node_cost(core.node_risk(0.5))
        pa, _ = core.astar_search(self.G, "Internet", "Internal_DB", cost)
        pd_, _ = core.dfs_search(self.G, "Internet", "Internal_DB")
        self.assertGreaterEqual(core.path_cost(pd_, cost), core.path_cost(pa, cost) - 1e-12)

    def test_draw_network_returns_figure(self):
        fig = core.draw_network(self.G, core.node_risk(0.5), ["Internet", "Edge_Router"], "t")
        self.assertTrue(hasattr(fig, "savefig"))


if __name__ == "__main__":
    unittest.main()
