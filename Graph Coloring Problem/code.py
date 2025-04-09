import networkx as nx
import matplotlib.pyplot as plt
import time

def load_col_file(filename):
    G = nx.Graph()
    
    with open(filename, "r") as file:
        for line in file:
            parts = line.split()
            print(parts)
            
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

def greedy_coloring(G):
    nodes_sorted = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)
    color_map = {}
    used_colors = set()

    print("\n********** Orden de los nodos y sus grados **********\n")
    
    for node in nodes_sorted:
        degree = G.degree(node)
        print(f"Nodo {node}: {degree} vecinos")

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

# Iniciar medición de tiempo total
start_total = time.time()

# Cargar el grafo desde archivo. NOTA: Cambiar el directorio por donde se encuentre en el archivo de la instancia.
graph = load_col_file("DIMACS/queen7_7.col")

# Mostrar información del grafo
print("\n********** Información del Grafo **********\n")
print(graph, "\n")
print("Nodos:", graph.nodes(), "\n")
print("Aristas:", graph.edges(), "\n")

# Ejecutar algoritmo de coloreo greedy
coloring_result, used_colors = greedy_coloring(graph)

# Mostrar resultados de coloreo
print("\n********** Colores asignados **********\n")
for node, color in coloring_result.items():
    print(f"Nodo {node}: {color}")

print(f"\nSe utilizaron {len(used_colors)} colores diferentes.")
print("Colores utilizados:", sorted(used_colors))

# Verificar si el coloreo es válido
if es_coloreo_valido(graph, coloring_result):
    print("\nEl coloreo es válido.")
else:
    print("\nEl coloreo NO es válido.")

# Dibujar el grafo y mostrar tiempo total justo antes del gráfico
draw_colored_graph(graph, coloring_result, start_total)
 # type: ignore