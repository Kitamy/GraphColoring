import networkx as nx
import matplotlib.pyplot as plt
import os
import time
from collections import deque
import random


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


def load_col_file(filename):
    G = nx.Graph()
    
    with open(filename, "r") as file:
        for line in file:
            parts = line.split()
            #print(parts)
            
            if not parts or parts[0] == 'c':
                continue
            elif parts[0] == 'p':
                num_nodes, num_edges = int(parts[2]), int(parts[3])
                G.add_nodes_from(range(1, num_nodes + 1))
            elif parts[0] == 'e':
                u, v = int(parts[1]), int(parts[2])
                G.add_edge(u, v)

    return G

color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]

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


# def count_conflicts(G, coloring):
#     conflicts = 0
#     for u, v in G.edges():
#         if coloring[u] == coloring[v]:
#             conflicts += 1
#     return conflicts



# def tabu_search_coloring(G: nx.Graph, initial_coloring, max_iter=1000, tabu_tenure=7):
#     current_coloring = initial_coloring.copy()
#     best_coloring = current_coloring.copy()
#     best_num_colors = len(set(best_coloring.values()))
    
#     tabu_list = deque(maxlen=tabu_tenure)
    
#     for _ in range(max_iter):
#         neighbors = []
#         used_colors = list(set(current_coloring.values()))

#         # Generar vecinos válidos: cambiar el color de un nodo a otro posible
#         for node in G.nodes():
#             current_color = current_coloring[node]
#             neighbor_colors = {current_coloring[n] for n in G.neighbors(node)}
            
#             for new_color in used_colors:
#                 if new_color != current_color and new_color not in neighbor_colors:
#                     move = (node, current_color, new_color)
#                     if move not in tabu_list:
#                         new_coloring = current_coloring.copy()
#                         new_coloring[node] = new_color
#                         neighbors.append((new_coloring, move))

#         if not neighbors:
#             break  # No hay vecinos válidos

#         # Evaluar vecinos: elegir el que tenga menos colores
#         #best_neighbor, best_move = min(neighbors, key=lambda x: len(set(x[0].values())))
#         best_neighbor, best_move = min(neighbors, key=lambda x: count_conflicts(G, x[0]))

#         current_coloring = best_neighbor
#         tabu_list.append(best_move)

#         current_colors_used = len(set(current_coloring.values()))
#         if current_colors_used < best_num_colors:
#             best_coloring = current_coloring.copy()
#             best_num_colors = current_colors_used

#     return best_coloring

def count_conflicts(G, coloring):
    """Cuenta cuántas aristas tienen nodos con el mismo color."""
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])

def random_coloring(G, color_count):
    """Asigna a cada nodo un color aleatorio entre los disponibles."""
    colors = [f"Color-{i}" for i in range(color_count)]
    return {node: random.choice(colors) for node in G.nodes()}

def tabu_search_coloring(G: nx.Graph, initial_coloring, max_iter=1000, tabu_tenure=10, restart_threshold=100):
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
    

def kempe_chain_swap(G: nx.Graph, coloring: dict, col1: str, col2: str, seed: int) -> dict:
    """
    Construye la componente conexa inducida por vértices de color col1 o col2
    a la que pertenece 'seed' y luego intercambia col1<->col2 en esa cadena.
    """
    # 1) Identificar la cadena de Kempe
    chain = set()
    queue = deque([seed])
    while queue:
        u = queue.popleft()
        if u in chain:
            continue
        if coloring[u] not in (col1, col2):
            continue
        chain.add(u)
        for v in G.neighbors(u):
            if v not in chain and coloring[v] in (col1, col2):
                queue.append(v)

    # 2) Hacer el swap en la cadena
    new_coloring = coloring.copy()
    for u in chain:
        new_coloring[u] = col2 if coloring[u] == col1 else col1
    return new_coloring

def local_search_with_kempe(G: nx.Graph, coloring: dict, max_iter: int = 1000) -> dict:
    """
    Aplica iterativamente swaps de Kempe para tratar de reducir el total de colores.
    En cada iteración:
      - Elige aleatoriamente dos colores col1, col2.
      - Escoge un seed (vértice) de color col1.
      - Realiza el swap de Kempe.
      - Si el nuevo coloreo mantiene la validez y reduce el número de colores, lo acepta.
    """
    best = coloring.copy()
    best_used = set(best.values())
    for it in range(max_iter):
        cols = list(best_used)
        if len(cols) < 2:
            break
        # 1) Elige dos colores al azar
        col1, col2 = random.sample(cols, 2)
        # 2) Escoge un vértice semilla de col1
        seeds = [v for v,c in best.items() if c == col1]
        if not seeds:
            continue
        seed = random.choice(seeds)
        # 3) Aplica Kempe-swap
        candidate = kempe_chain_swap(G, best, col1, col2, seed)
        # 4) Verifica validez
        if not es_coloreo_valido(G, candidate):
            continue
        # 5) Comprueba si podemos eliminar algún color
        used = set(candidate.values())
        # si hay menos colores, lo aceptamos
        if len(used) < len(best_used):
            best = candidate
            best_used = used
            print(f"Iter {it}: reducido a {len(best_used)} colores ({col1}<->{col2})")
            # resetear iteraciones para seguir explorando ese nuevo nivel
            it = 0

    return best




def draw_colored_graph(G, coloring_result, start_time):
    plt.figure(figsize=(12, 8))
    
    unique_colors = list(set(coloring_result.values()))
    color_mapping = {color: idx for idx, color in enumerate(unique_colors)}
    node_colors = [color_mapping[coloring_result[node]] for node in G.nodes()]

    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=10, edge_color="gray",
        cmap=plt.cm.get_cmap("tab10")
    )

    plt.title("Grafo coloreado")

    # Mostrar tiempo justo antes de mostrar el grafo
    end_total = time.time()
    print(f"\nTiempo total de ejecución: {end_total - start_time:.4f} segundos")

    plt.show()

def es_coloreo_valido(G, coloring):
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            print(f"Error: Los nodos {u} y {v} tienen el mismo color ({coloring[u]})")
            return False
    return True



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





# Iniciar medición de tiempo total
start_total = time.time()

# Cargar el grafo desde archivo. NOTA: Cambiar el directorio por donde se encuentre en el archivo de la instancia.
graph = load_col_file("DIMACS/queen11_11.col")

print("\n********** Información del Grafo **********\n")
print(graph, "\n")
# print("Nodos:", graph.nodes(), "\n")
# print("Aristas:", graph.edges(), "\n")

# Ejecutar algoritmo de coloreo greedy
#coloring_result, used_colors = greedy_coloring(graph)

coloring_result, used_colors = dsatur_coloring(graph)

# Mostrar información del grafo
print("\n********** Información del Grafo **********\n")
print(f"Número de nodos: {graph.number_of_nodes()}")
print(f"Número de aristas: {graph.number_of_edges()}")

# Mostrar resultados de coloreo
print("\n********** Colores asignados **********\n")
# for node, color in coloring_result.items():
#     print(f"Nodo {node}: {color}")

print(f"\nSe utilizaron {len(used_colors)} colores diferentes.")
print("Colores utilizados:", sorted(used_colors))

# Verificar si el coloreo es válido
if es_coloreo_valido(graph, coloring_result):
    print("\nEl coloreo es válido.")
else:
    print("\nEl coloreo NO es válido.")


 # Aplicar búsqueda local para mejorar el coloreo
# coloring_result = local_search_improvement(graph, coloring_result)
# used_colors = set(coloring_result.values())

# print("\n********** Colores después de la mejora **********\n")
# print("Numero de colores usados después de la mejora:", len(used_colors))
# print("Colores utilizados después de la mejora:", sorted(used_colors))

# Dibujar el grafo y mostrar tiempo total justo antes del gráfico
#draw_colored_graph(graph, coloring_result, start_total)
 # type: ignore


# Aplicar DSATUR
# dsatur_result, _ = dsatur_coloring(graph)
# # Aplicar Tabu Search
# #coloring_result = tabu_search_coloring(graph, coloring_result, max_iter=500, tabu_tenure=10)
# tabu_result = tabu_search_coloring(graph, dsatur_result, max_iter=1000, tabu_tenure=15, restart_threshold=100)
# used_colors = set(coloring_result.values())

# print(f"\nColores después de Tabu Search: {len(used_colors)}")


# Después de obtener un coloreo inicial (p. ej. DSATUR):
initial_coloring, _ = dsatur_coloring(graph)
improved = local_search_with_kempe(graph, initial_coloring, max_iter=5000)

print(f"Colores antes: {len(set(initial_coloring.values()))}")
print(f"Colores después de Kempe swaps: {len(set(improved.values()))}")


def run_all_instances(folder="DIMACS", use_tabu=False):
    header = f"{'Instance':<20} {'Greedy':<10} {'DSATUR':<10}  {'Optimal':<10} {'Greedy Gap':<12} {'DSATUR Gap':<12}"
    print(header)
    print("-" * len(header))

    for filename in os.listdir(folder):
        if not filename.endswith(".col"):
            continue

        filepath = os.path.join(folder, filename)
        G = load_col_file(filepath)

        # GREEDY
        greedy_result, greedy_colors = greedy_coloring(G)
        greedy_count = len(greedy_colors)

        # DSATUR
        dsatur_result, dsatur_colors = dsatur_coloring(G)
        dsatur_count = len(dsatur_colors)

        # Óptimo conocido
        optimal = BEST_KNOWN_COLORS.get(filename, "?")

        # TABU solo si dsatur_count > optimal
        # if isinstance(optimal, int) and dsatur_count > optimal:
        #     tabu_result = tabu_search_coloring(
        #         G,
        #         dsatur_result,
        #         max_iter=1000,
        #         tabu_tenure=15,
        #         restart_threshold=100
        #     )
        #     tabu_count = len(set(tabu_result.values()))
        # else:
        #     # No ejecutar tabu, usar el mismo resultado que DSATUR
        #     tabu_count = dsatur_count

        def gap(alg_value):
            if isinstance(optimal, int):
                return f"{((alg_value - optimal) / optimal) * 100:.2f}%"
            else:
                return "N/A"

        print(f"{filename:<20} {greedy_count:<10} {dsatur_count:<10}  {optimal:<10} {gap(greedy_count):<12} {gap(dsatur_count):<12}")




#run_all_instances()
#run_all_instances("QUEENS")
#run_all_instances("DIMACS")