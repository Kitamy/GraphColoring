#Pruebas
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
color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]

# Coloreo Greedy
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

# Búsqueda local con intercambio de colores
def local_search_color_swap(G, color_map):
    improvement_found = False
    # Obtener los nodos y sus colores actuales
    node_color_map = {node: color_map[node] for node in G.nodes()}

    for node in G.nodes():
        current_color = node_color_map[node]
        # Buscar vecinos con el mismo color
        same_color_neighbors = [neighbor for neighbor in G.neighbors(node) if node_color_map[neighbor] == current_color]
        
        for neighbor in same_color_neighbors:
            # Intentar intercambiar colores entre el nodo y el vecino
            neighbor_color = node_color_map[neighbor]
            node_color_map[node], node_color_map[neighbor] = neighbor_color, current_color

            # Verificar si la solución es válida
            valid_swap = all(node_color_map[neighbor] != node_color_map[adj] for adj in G.neighbors(neighbor))
            if valid_swap:
                improvement_found = True
                color_map[node] = node_color_map[node]
                color_map[neighbor] = node_color_map[neighbor]
                break  # Si encontramos una mejora, salimos del bucle

            # Si no es válido, revertir el intercambio
            node_color_map[node], node_color_map[neighbor] = current_color, neighbor_color

        if improvement_found:
            break  # Si encontramos una mejora, terminamos el ciclo

    # Devuelve la nueva asignación de colores y si se encontró una mejora
    new_used_colors = set(color_map.values())
    return color_map, improvement_found

# Búsqueda local: primera mejora
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

# ------------ Iterated Local Search (ILS) Mejorado ------------
def perturb_solution(color_map, perturbation_rate=0.20):
    new_map = color_map.copy()
    nodes = list(new_map.keys())
    n = int(len(nodes) * perturbation_rate)  # Aumentamos la tasa de perturbación
    to_perturb = random.sample(nodes, n)

    # Perturbación más controlada: elegir colores ya usados
    available_colors = list(set(color_map.values()))
    for node in to_perturb:
        new_map[node] = random.choice(available_colors)  # Reasigna el color de manera más controlada

    return new_map

def iterated_local_search(G, initial_map, iterations=100, local_max_iterations=100000):
    current_map = initial_map.copy()
    best_map = current_map.copy()
    best_color_count = len(set(current_map.values()))

    for _ in range(iterations):
        # Realiza más de una perturbación para evitar quedarse atrapado
        for _ in range(3):  # Realiza 3 perturbaciones por ciclo
            perturbed_map = perturb_solution(current_map, perturbation_rate=0.30)  # Perturbación más fuerte
            improved_map, _ = local_search_first_improvement(G, perturbed_map, max_iterations=local_max_iterations)
            current_color_count = len(set(improved_map.values()))

            # Si la solución perturbada es mejor, la usamos
            if current_color_count < best_color_count:
                best_color_count = current_color_count
                best_map = improved_map.copy()

        current_map = best_map.copy()

    return best_map, best_color_count


# ------------ Algoritmo DSATUR ------------
def dsatur_coloring(G: nx.Graph) -> tuple[dict, set]:
    # Inicializaciones
    nodes = set(G.nodes())
    color_map: dict = {}
    # Colores disponibles (puedes usar tu lista color_names global)
    colors = []
    # Saturación: para cada nodo, el conjunto de colores en sus vecinos
    saturation: dict = {u: set() for u in nodes}
    # Grados originales
    degree = dict(G.degree())

    while nodes:
        # Elegir el nodo con mayor saturación; empata con mayor grado
        u = max(
            nodes,
            key=lambda x: (len(saturation[x]), degree[x])
        )

        # Determinar el color mínimo que no use ningún vecino
        neighbor_colors = { color_map[v] for v in G.neighbors(u) if v in color_map }
        # Encuentra el primer índice de color libre
        for idx, c in enumerate(colors):
            if c not in neighbor_colors:
                chosen_color = c
                break
        else:
            # Si ningún color existente vale, creamos uno nuevo
            chosen_color = f"Color-{len(colors)}"
            colors.append(chosen_color)

        # Asignar color
        color_map[u] = chosen_color
        nodes.remove(u)

        # Actualizar saturación de vecinos no coloreados
        for v in G.neighbors(u):
            if v in nodes:
                saturation[v].add(chosen_color)

    return color_map, set(colors)


# ------------ Validador  ------------
def es_coloreo_valido(G: nx.Graph, color_map: dict) -> bool:
    """
    Comprueba que para cada arista (u, v) en G, color_map[u] != color_map[v].
    Devuelve True si es válido, False en caso contrario.
    """
    for u, v in G.edges():
        if color_map.get(u) == color_map.get(v):
            return False
    return True

def conflictos_invalidez(G: nx.Graph, color_map: dict, max_mostrar: int = 10):
    """
    Devuelve una lista de hasta max_mostrar pares (u, v) que violan la validez.
    """
    conflictos = [(u, v) for u, v in G.edges() if color_map.get(u) == color_map.get(v)]
    return conflictos[:max_mostrar]


# ------------ Configuración ------------
@dataclass
class Config:
    use_greedy: bool = True
    use_dsatur: bool = True
    use_local: bool = True
    use_ils: bool = True  # Agregar la opción para usar ILS
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

    def ils(self, G, init_coloring):
        # Utiliza la solución de Greedy como inicialización
        start = time.time()
        result, count = iterated_local_search(G, init_coloring)
        elapsed = time.time() - start
        return result, count, elapsed
    
    def dsatur(self, G):
        start = time.time()
        coloring, used = dsatur_coloring(G)
        elapsed = time.time() - start
        return coloring, len(used), elapsed

# ------------ Benchmark ------------
class GraphColoringBenchmark:
    def __init__(self, config: Config):
        self.cfg = config
        self.solver = GraphColoringSolver(config)

    def run_all(self, folder: str):
        cols = ['Instance']
        if self.cfg.use_greedy: cols += ['Greedy' + (' (t)' if self.cfg.time_enabled else '')]
        if self.cfg.use_dsatur: cols += ['DSATUR']
        if self.cfg.use_local:  cols += ['Local'  + (' (t)' if self.cfg.time_enabled else ''), 'Local Improved']
        if self.cfg.use_ils:    cols += ['ILS'    + (' (t)' if self.cfg.time_enabled else ''), 'ILS Improved']
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
                # Verificar factibilidad:
                if not es_coloreo_valido(G, cmap_g):
                    print(f"[ERROR] Solución Greedy para {fname} NO es válida. Conflictos:", 
                        conflictos_invalidez(G, cmap_g))
                    row.append("INVÁLIDA")
                    row.append("-")
                    base_map = {}
                else:
                    row.append(f"{g_cnt}{f' ({g_t:.3f}s)' if self.cfg.time_enabled else ''}")
                base_map = cmap_g
            else:
                base_map = {}

            if self.cfg.use_dsatur:
                cmap_d, d_cnt, _ = self.solver.dsatur(G)
                # Verificar factibilidad:
                if not es_coloreo_valido(G, cmap_d):
                    print(f"[ERROR] Solución DSATUR para {fname} NO es válida. Conflictos:", 
                        conflictos_invalidez(G, cmap_d))
                    row.append("INVÁLIDA")
                    row.append("-")
                else:
                    row.append(f"{d_cnt}")

            if self.cfg.use_local:
                cmap_l, cmap_cnt, local_improved, l_t = self.solver.local(G, base_map)
                # Verificar factibilidad:
                if not es_coloreo_valido(G, cmap_l):
                    print(f"[ERROR] Solución Local para {fname} NO es válida. Conflictos:", 
                        conflictos_invalidez(G, cmap_l))
                    row.append("INVÁLIDA")
                    row.append("-")
                else:
                    row.append(f"{cmap_cnt}{f' ({l_t:.3f}s)' if self.cfg.time_enabled else ''}")
                    row.append("Yes" if local_improved and cmap_cnt < g_cnt else "No")
                base_map = cmap_l if local_improved else base_map



            if self.cfg.use_ils:
                cmap_ils, cmap_ils_cnt, ils_t = self.solver.ils(G, base_map)  # ILS usando la solución Greedy
                # Verificar factibilidad:
                if not es_coloreo_valido(G, cmap_ils):
                    print(f"[ERROR] Solución ILS para {fname} NO es válida. Conflictos:", 
                        conflictos_invalidez(G, cmap_ils))
                    # Decides si usas la solución inicial o marcas como inválida:
                    row.append("INVÁLIDA")
                    row.append("-")
                else:
                    row.append(f"{cmap_ils_cnt}{f' ({ils_t:.3f}s)' if self.cfg.time_enabled else ''}")
                    row.append("Yes" if cmap_ils_cnt < g_cnt else "No")

            row.append(str(opt) if opt is not None else "?")
            print("  ".join(f"{r:<16}" for r in row))

# ------------ Uso ------------
if __name__ == '__main__':
    cfg = Config(use_greedy=True, use_local=True, use_ils=False, time_enabled=True)
    bench = GraphColoringBenchmark(cfg)
    bench.run_all('DIMACS')
