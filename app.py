import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Cybersecurity Agent", layout="wide")
st.title("🛡️ AI-Based Cybersecurity Monitoring & Attack Path Prediction")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("1. Upload Network Data")
uploaded_file = st.sidebar.file_uploader("Upload CICIDS2017 CSV", type=["csv"])

if uploaded_file is not None:
    # 1. Load and Train ML Model
    df = pd.read_csv(uploaded_file, nrows=5000)
    df.columns = df.columns.str.strip()
    target_col = df.columns[-1]
    df['Is_Attack'] = df[target_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    
    features = df.select_dtypes(include=['float64', 'int64']).columns
    features = [f for f in features if f != 'Is_Attack']
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['Is_Attack']

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    st.sidebar.success("✅ Data loaded & AI Model Trained!")

    # --- MAIN DASHBOARD ---
    st.subheader("2. Real-Time Threat Detection")
    row_idx = st.slider("Select Network Flow to Analyze", 0, len(df)-1, 0)
    prediction = model.predict(X.iloc[[row_idx]])[0]
    
    col1, col2 = st.columns(2)
    col1.metric("Status", "🚨 MALICIOUS" if prediction == 1 else "✅ BENIGN")
    col2.metric("Flow Duration", f"{df.iloc[row_idx]['Flow Duration']}")

    # --- SEARCH ALGORITHM COMPARISON ---
    st.subheader("3. AI Agent: Search Algorithm Comparison (Section 3.1 & 6.4)")
    
    # Define Complex Corporate Network
    G = nx.DiGraph()
    nodes = ['Internet', 'Edge_Router', 'WAF', 'DMZ_Web', 'DMZ_App', 
             'Core_Switch', 'HR_Workstation', 'Dev_Workstation', 'Internal_DB', 'Backup_Server']
    G.add_nodes_from(nodes)
    edges = [
        ('Internet', 'Edge_Router'), ('Edge_Router', 'WAF'),
        ('WAF', 'DMZ_Web'), ('WAF', 'DMZ_App'),
        ('DMZ_Web', 'Core_Switch'), ('DMZ_App', 'Core_Switch'),
        ('Core_Switch', 'HR_Workstation'), ('Core_Switch', 'Dev_Workstation'),
        ('HR_Workstation', 'Internal_DB'), ('Dev_Workstation', 'Internal_DB'),
        ('HR_Workstation', 'Backup_Server'), ('Dev_Workstation', 'Backup_Server')
    ]
    G.add_edges_from(edges)

    # Custom BFS to count explored nodes
    def bfs_search(G, start, goal):
        explored = set()
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == goal: return path, len(explored) + 1
            if node not in explored:
                explored.add(node)
                for neighbor in G.neighbors(node):
                    queue.append(list(path) + [neighbor])
        return None, len(explored)

    # Custom DFS to count explored nodes
    def dfs_search(G, start, goal):
        explored = set()
        stack = [[start]]
        while stack:
            path = stack.pop()
            node = path[-1]
            if node == goal: return path, len(explored) + 1
            if node not in explored:
                explored.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in explored:
                        stack.append(list(path) + [neighbor])
        return None, len(explored)

    # A* Heuristic (ML Risk Scores)
    def ml_heuristic(node, goal):
        risk_map = {'DMZ_Web': 0.8, 'HR_Workstation': 0.9, 'Dev_Workstation': 0.2,
                    'Internal_DB': 0.7, 'Core_Switch': 0.1, 'WAF': 0.05,
                    'Edge_Router': 0.1, 'DMZ_App': 0.6, 'Backup_Server': 0.4, 'Internet': 0.0}
        return 1.0 - risk_map.get(node, 0.5)

    start, goal = 'Internet', 'Internal_DB'

    # Run Algorithms and measure time
    t0 = time.time(); path_bfs, exp_bfs = bfs_search(G, start, goal); t_bfs = time.time() - t0
    t0 = time.time(); path_dfs, exp_dfs = dfs_search(G, start, goal); t_dfs = time.time() - t0
    t0 = time.time(); path_astar = nx.astar_path(G, start, goal, heuristic=ml_heuristic); t_astar = time.time() - t0

    # Display Comparison Table (Crucial for Section 6.4)
    st.write("**📊 Algorithm Performance Comparison:**")
    comparison_data = {
        'Algorithm': ['BFS (Uninformed)', 'DFS (Uninformed)', 'A* Search (AI-Informed)'],
        'Path Found': [' ➔ '.join(path_bfs), '  '.join(path_dfs), ' ➔ '.join(path_astar)],
        'Nodes Explored': [exp_bfs, exp_dfs, 'Optimal (Heuristic guided)'],
        'Execution Time (s)': [f"{t_bfs:.6f}", f"{t_dfs:.6f}", f"{t_astar:.6f}"]
    }
    st.table(pd.DataFrame(comparison_data))

    # Interactive Visualization
    st.write("** Visualize Selected Algorithm Path:**")
    algo_choice = st.radio("Select Algorithm to Visualize:", ('A* Search (AI Agent)', 'BFS (Breadth-First)', 'DFS (Depth-First)'), horizontal=True)
    
    if algo_choice == 'A* Search (AI Agent)': path_to_draw = path_astar
    elif algo_choice == 'BFS (Breadth-First)': path_to_draw = path_bfs
    else: path_to_draw = path_dfs

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=9, ax=ax, arrows=True, edge_color='gray')
    
    path_edges = list(zip(path_to_draw[:-1], path_to_draw[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=path_to_draw, node_color='red', node_size=2500, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in path_to_draw}, font_color='white', font_weight='bold', ax=ax)
    
    plt.title(f"Visualizing {algo_choice} Path (Red)", fontsize=14)
    st.pyplot(fig)

else:
    st.info("👈 Please upload a CICIDS2017 CSV file in the sidebar to start the AI Agent.")