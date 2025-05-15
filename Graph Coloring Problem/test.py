import networkx as nx
import os
import time
import random
from dataclasses import dataclass

# ------------ Instancias conocidas ------------
BEST_KNOWN_COLORS = {
    "david.col": 11,
    "games120.col": 9,
    "huck.col": 11,
    "le450_5a.col": 5,
    "miles250.col": 8,
    "miles500.col": 20,
    "miles1500.col": 73,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "queen6_6.col": 7,
    "queen7_7.col": 7,
    "queen8_8.col": 9,
    "queen8_12.col": 12,
    "queen11_11.col": 11,
}

# ------------ Carga de grafos ------------
def load_col_file(path: str) -> nx.Graph:
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == 'c':
                continue
            if parts[0] == 'p':
                n, m = int(parts[2]), int(parts[3])
                G.add_nodes_from(range(1, n+1))
            elif parts[0] == 'e':
                u, v = map(int, parts[1:3])
                G.add_edge(u, v)
    return G

# ------------ Algoritmos de coloreo ------------
color_names = [
    "Red", "Green", "Blue", "Yellow", "Orange", "Purple",
    "Pink", "Brown", "Gray", "Cyan"
]

def greedy_coloring(G: nx.Graph) -> tuple[dict, set]:
    nodes_sorted = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)
    color_map = {}
    used_colors = set()

    for node in nodes_sorted:
        neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(node) if neighbor in color_map}
        color_index = 0
        while color_index < len(color_names) and color_names[color_index] in neighbor_colors:
            color_index += 1

        if color_index >= len(color_names):
            new_color = f"Color-{color_index}"
            color_names.append(new_color)
        else:
            new_color = color_names[color_index]

        color_map[node] = new_color
        used_colors.add(new_color)

    return color_map, used_colors

def local_search_first_improvement(G, color_map, max_iterations=100000):
    improvement_found = False
    for _ in range(max_iterations):
        used_colors = set(color_map.values())

        for node in G.nodes():
            current_color = color_map[node]
            neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(node)}

            for candidate_color in sorted(used_colors):
                if candidate_color == current_color:
                    continue
                if candidate_color not in neighbor_colors:
                    color_map[node] = candidate_color
                    improvement_found = True
                    break

            if improvement_found:
                break

        new_used_colors = {color_map[n] for n in G.nodes()}
        if not improvement_found or len(new_used_colors) >= len(used_colors):
            break

    return color_map, improvement_found

# ------------ Backtracking con poda ------------
def backtracking_coloring(G: nx.Graph, max_colors: int) -> tuple[dict, float]:
    nodes = list(G.nodes())
    color_map = {}
    start = time.time()

    def is_valid(node, color):
        return all(color_map.get(neigh) != color for neigh in G.neighbors(node))

    def solve(index):
        if index == len(nodes):
            return True
        for c in range(max_colors):
            if is_valid(nodes[index], c):
                color_map[nodes[index]] = c
                if solve(index + 1):
                    return True
                del color_map[nodes[index]]
        return False

    # Intentamos colorear con k colores para k = 1 .. max_colors
    for k in range(1, max_colors + 1):
        color_map.clear()
        if solve(0):
            elapsed = time.time() - start
            named_map = {n: f"Color-{color_map[n]}" for n in color_map}
            return named_map, elapsed

    return {}, time.time() - start

# ------------ Configuración ------------
@dataclass
class Config:
    use_greedy: bool = True
    use_local: bool = True
    use_backtracking: bool = True
    time_enabled: bool = False

# ------------ Solver ------------
class GraphColoringSolver:
    def __init__(self, config: Config):
        self.cfg = config

    def greedy(self, G):
        start = time.time()
        coloring, colors = greedy_coloring(G)
        elapsed = time.time() - start
        return coloring, len(colors), elapsed

    def local(self, G, init_coloring):
        start = time.time()
        result, improved = local_search_first_improvement(G, init_coloring)
        elapsed = time.time() - start
        return result, len(set(result.values())), improved, elapsed

    def backtracking(self, G, max_colors):
        result, elapsed = backtracking_coloring(G, max_colors=max_colors)
        return result, len(set(result.values())), elapsed

# ------------ Benchmark ------------
class GraphColoringBenchmark:
    def __init__(self, config: Config):
        self.cfg = config
        self.solver = GraphColoringSolver(config)

    def run_all(self, folder: str):
        backtrack_files = {
            "queen6_6.col",
            "queen7_7.col",
            "queen8_8.col",
            "queen8_12.col",
            "queen11_11.col",
        }

        cols = ['Instance']
        if self.cfg.use_greedy: cols += ['Greedy' + (' (t)' if self.cfg.time_enabled else '')]
        if self.cfg.use_local:  cols += ['Local'  + (' (t)' if self.cfg.time_enabled else ''), 'Local Improved']
        if self.cfg.use_backtracking: cols += ['Backtrack (t)']
        cols += ['Optimal']

        print("  ".join(f"{c:<16}" for c in cols))
        print('-' * (len(cols) * 14))

        for fname in os.listdir(folder):
            if not fname.endswith('.col'): continue
            G = load_col_file(os.path.join(folder, fname))
            opt = BEST_KNOWN_COLORS.get(fname)
            row = [fname]

            if self.cfg.use_greedy:
                cmap_g, g_cnt, g_t = self.solver.greedy(G)
                row.append(f"{g_cnt}{f' ({g_t:.3f}s)' if self.cfg.time_enabled else ''}")
                base_map = cmap_g
            else:
                base_map = {}

            if self.cfg.use_local:
                cmap_l, cmap_cnt, local_improved, l_t = self.solver.local(G, base_map)
                row.append(f"{cmap_cnt}{f' ({l_t:.3f}s)' if self.cfg.time_enabled else ''}")
                row.append("Yes" if local_improved and cmap_cnt < g_cnt else "No")
                base_map = cmap_l if local_improved else base_map

            if self.cfg.use_backtracking and fname in backtrack_files:
                max_colors = len(set(base_map.values())) if base_map else len(G.nodes())
                cmap_bt, bt_cnt, bt_t = self.solver.backtracking(G, max_colors=max_colors)
                row.append(f"{bt_cnt}{f' ({bt_t:.3f}s)' if self.cfg.time_enabled else ''}")
            elif self.cfg.use_backtracking:
                row.append("Skipped")

            row.append(str(opt) if opt is not None else "?")
            print("  ".join(f"{r:<16}" for r in row))

# ------------ Uso ------------
if __name__ == '__main__':
    cfg = Config(use_greedy=True, use_local=True, use_backtracking=True, time_enabled=True)
    bench = GraphColoringBenchmark(cfg)
    bench.run_all('DIMACS')
