import networkx as nx
import os
import time
import random
from collections import deque
from dataclasses import dataclass

# ------------ Instancias conocidas ------------
BEST_KNOWN_COLORS = {
    "david.col": 11,
    "fpsol2.i.1.col": 65,
    "games120.col": 9,
    "huck.col": 11,
    "inithx.i.1.col": 54,
    "le450_5a.col": 5,
    "le450_5c.col": 5,
    "le450_15b.col": 15,
    "le450_25a.col": 25,
    "le450_25c.col": 25,
    "le450_25d.col": 25,
    "miles250.col": 8,
    "miles500.col": 20,
    "miles1500.col": 73,
    "mulsol.i.1.col": 49,
    "mulsol.i.5.col": 31,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "queen6_6.col": 7,
    "queen7_7.col": 7,
    "queen8_8.col": 9,
    "queen8_12.col": 12,
    "queen10_10.col": 10,
    "queen11_11.col": 11,
    "zeroin.i.1.col": 49,
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
color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]


# ------------ Greedy Coloring ------------
def greedy_coloring(G: nx.Graph) -> tuple[dict, set]:
    nodes_sorted = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)
    color_map = {}
    used_colors = set()

    #print("\n********** Orden de los nodos y sus grados **********\n")
    
    for node in nodes_sorted:
        degree = G.degree(node)
        #print(f"Nodo {node}: {degree} vecinos")

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


# ------------ DSATUR Coloring ------------
def dsatur_coloring(G: nx.Graph) -> tuple[dict, set]:
    color_map = {}
    saturation = {node: 0 for node in G.nodes()}
    uncolored = set(G.nodes())
    degrees = dict(G.degree())

    while uncolored:
        # Elegir el nodo con mayor saturación (y mayor grado en caso de empate)
        next_node = max(
            uncolored,
            key=lambda node: (saturation[node], degrees[node])
        )

        neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(next_node) if neighbor in color_map}

        # Buscar el primer color disponible
        for i, color in enumerate(color_names):
            if color not in neighbor_colors:
                assigned_color = color
                break
        else:
            # Crear color nuevo si se necesitan más
            assigned_color = f"Color-{len(color_names)}"
            color_names.append(assigned_color)

        color_map[next_node] = assigned_color
        uncolored.remove(next_node)

        # Actualizar saturación de sus vecinos no coloreados
        for neighbor in G.neighbors(next_node):
            if neighbor in uncolored:
                neighbor_used_colors = {color_map[n] for n in G.neighbors(neighbor) if n in color_map}
                saturation[neighbor] = len(neighbor_used_colors)

    used_colors = set(color_map.values())
    return color_map, used_colors


# ------------ Algoritmos de búsqueda ------------

# ------------ Tabu Search ------------
def count_conflicts(G, coloring):
    """Cuenta cuántas aristas tienen nodos con el mismo color."""
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])

def random_coloring(G, color_count):
    """Asigna a cada nodo un color aleatorio entre los disponibles."""
    colors = [f"Color-{i}" for i in range(color_count)]
    return {node: random.choice(colors) for node in G.nodes()}

def tabu_search_coloring(G: nx.Graph, initial_coloring, max_iter=10, tabu_tenure=10, restart_threshold=100):
    current_coloring = initial_coloring.copy()
    best_coloring = current_coloring.copy()
    best_conflicts = count_conflicts(G, current_coloring)

    tabu_list = deque(maxlen=tabu_tenure)
    no_improve_count = 0

    for i in range(max_iter):
        neighbors = []
        used_colors = list(set(current_coloring.values()))

        for node in G.nodes():
            current_color = current_coloring[node]
            neighbor_colors = {current_coloring[n] for n in G.neighbors(node)}
            
            for new_color in used_colors:
                if new_color == current_color:
                    continue
                if new_color not in neighbor_colors:  # solo movimientos válidos
                    move = (node, current_color, new_color)
                    if move not in tabu_list:
                        new_coloring = current_coloring.copy()
                        new_coloring[node] = new_color
                        neighbors.append((new_coloring, move))

        if not neighbors:
            break

        # Evaluar por número de conflictos
        best_neighbor, best_move = min(neighbors, key=lambda x: count_conflicts(G, x[0]))
        current_coloring = best_neighbor
        tabu_list.append(best_move)

        current_conflicts = count_conflicts(G, current_coloring)
        if current_conflicts < best_conflicts:
            best_coloring = current_coloring.copy()
            best_conflicts = current_conflicts
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Reinicio si no mejora tras muchas iteraciones
        if no_improve_count >= restart_threshold:
            # Reinicia con una solución aleatoria con mismo número de colores usados
            current_coloring = random_coloring(G, len(set(current_coloring.values())))
            no_improve_count = 0
            # También podrías reiniciar con dsatur_coloring(G)[0] si prefieres
            # current_coloring = dsatur_coloring(G)[0]

    # Como usamos conflicto como criterio, devolver solución válida más cercana
    if count_conflicts(G, best_coloring) == 0:
        return best_coloring
    else:
        print("Advertencia: Tabu terminó con conflictos")
        return best_coloring
    

# ------------ Local Search ------------

def local_search_improvement(G, color_map, max_iterations=100000):
    for _ in range(max_iterations):
        used_colors = set(color_map.values())
        changed = False

        for node in G.nodes():
            current_color = color_map[node]
            neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(node)}

            for candidate_color in sorted(used_colors):
                if candidate_color == current_color:
                    continue
                if candidate_color not in neighbor_colors:
                    color_map[node] = candidate_color
                    changed = True
                    break

        # Recalcular colores usados
        new_used_colors = {color_map[n] for n in G.nodes()}
        if len(new_used_colors) >= len(used_colors):
            break  # No hay mejora real
    return color_map



@dataclass
class Config:
    use_greedy: bool = True
    use_dsatur: bool = True
    use_tabu: bool = False
    use_local: bool = False
    time_enabled: bool = False

class GraphColoringSolver:
    def __init__(self, config: Config):
        self.cfg = config

    def greedy(self, G):
        start = time.time()
        coloring, colors = greedy_coloring(G)
        elapsed = time.time() - start
        return coloring, len(colors), elapsed

    def dsatur(self, G):
        start = time.time()
        coloring, colors = dsatur_coloring(G)
        elapsed = time.time() - start
        return coloring, len(colors), elapsed

    def tabu(self, G, init_coloring):
        start = time.time()
        result = tabu_search_coloring(G, init_coloring)
        elapsed = time.time() - start
        return result, len(set(result.values())), elapsed

    def local(self, G, init_coloring):
        start = time.time()
        result = local_search_improvement(G, init_coloring)
        elapsed = time.time() - start
        return result, len(set(result.values())), elapsed

class GraphColoringBenchmark:
    def __init__(self, config: Config):
        self.cfg = config
        self.solver = GraphColoringSolver(config)

    def run_all(self, folder: str):
        # Construcción del encabezado dinámico
        cols = ['Instance']
        if self.cfg.use_greedy: cols += ['Greedy' + (' (t)' if self.cfg.time_enabled else '')]
        if self.cfg.use_dsatur: cols += ['DSATUR' + (' (t)' if self.cfg.time_enabled else '')]
        if self.cfg.use_tabu:   cols += ['Tabu'   + (' (t)' if self.cfg.time_enabled else '')]
        if self.cfg.use_local:  cols += ['Local'  + (' (t)' if self.cfg.time_enabled else '')]
        cols += ['Optimal']
        print("  ".join(f"{c:<16}" for c in cols))
        print('-' * (len(cols) * 14))

        for fname in os.listdir(folder):
            if not fname.endswith('.col'): continue
            G = load_col_file(os.path.join(folder, fname))
            opt=BEST_KNOWN_COLORS.get(fname)
            row = [fname]

            # Greedy
            if self.cfg.use_greedy:
                cmap_g,g_cnt,g_t=self.solver.greedy(G)
                row.append(f"{g_cnt}{f' ({g_t:.3f}s)' if self.cfg.time_enabled else ''}")
            # DSATUR
            if self.cfg.use_dsatur:
                cmap_d,ds_cnt,ds_t=self.solver.dsatur(G)
                row.append(f"{ds_cnt}{f' ({ds_t:.3f}s)' if self.cfg.time_enabled else ''}")
                base_map, base_cnt = cmap_d, ds_cnt
            else:
                base_map, base_cnt = cmap_g, g_cnt
            # Tabu solo si habilitado y base_cnt > óptimo
            if self.cfg.use_tabu:
                if isinstance(opt,int) and base_cnt>opt:
                    cmap_t, t_cnt, t_t=self.solver.tabu(G, base_map)
                    row.append(f"{t_cnt}{f' ({t_t:.3f}s)' if self.cfg.time_enabled else ''}")
                else:
                    row.append("-")
            # Local solo si habilitado y base_cnt > óptimo
            if self.cfg.use_local:
                if isinstance(opt,int) and base_cnt>opt:
                    cmap_l, cmap_cnt, l_t=self.solver.local(G, base_map)
                    row.append(f"{cmap_cnt}{f' ({l_t:.3f}s)' if self.cfg.time_enabled else ''}")
                else:
                    row.append("-")

            opt = BEST_KNOWN_COLORS.get(fname, '?')
            row.append(str(opt))

            print("  ".join(f"{r:<16}" for r in row))



# ------------ Uso ------------
if __name__ == '__main__':
    # Ejemplo de configuración
    cfg = Config(use_greedy=True, use_dsatur=True, use_tabu=True, use_local=False, time_enabled=True)
    bench = GraphColoringBenchmark(cfg)
    bench.run_all('DIMACS')
