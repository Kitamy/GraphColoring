import networkx as nx
import matplotlib.pyplot as plt

def load_col_file(filename):
    G = nx.Graph()
    
    with open(filename, "r") as file:  # Abre el archivo a leer
        for line in file:  # Recorre las conexiones del archivo
            parts = line.split()  # Devuelve una lista con los fragmentos de la cadena
            print(parts)
            
            if not parts or parts[0] == 'c':  # Ignorar comentarios
                continue
                
            elif parts[0] == 'p':  # Si parts es igual a 'p', lee el número de nodos y aristas
                num_nodes, num_edges = int(parts[2]), int(parts[3])
                G.add_nodes_from(range(1, num_nodes + 1))  # Añadir nodos

            elif parts[0] == 'e':  # Si parts es igual a 'e', lee las aristas
                u, v = int(parts[1]), int(parts[2])
                G.add_edge(u, v)  # Añade las aristas (conexiones)

    return G

# Lista de colores predefinidos
color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]

def greedy_coloring(G):
    # Paso 1: Ordenar los nodos por grado en orden descendente
    nodes_sorted = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)

    # Diccionario para almacenar el color de cada nodo
    color_map = {}

    # Conjunto para almacenar los colores utilizados sin repetirlos
    used_colors = set()

    print("\n********** Orden de los nodos y sus grados **********\n")
    
    # Paso 2 y 3: Asignar colores a los nodos en el orden establecido
    for node in nodes_sorted:
        degree = G.degree(node)  # Obtener el número de vecinos del nodo
        print(f"Nodo {node}: {degree} vecinos")  # Imprimir cantidad de vecinos

        # Obtener colores de los vecinos ya coloreados
        neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(node) if neighbor in color_map}
        
        # Encontrar el primer color disponible de la lista
        color_index = 0
        while color_index < len(color_names) and color_names[color_index] in neighbor_colors:
            color_index += 1

        # Si se acaban los colores predefinidos, generar nuevos colores
        if color_index >= len(color_names):
            new_color = f"Color-{color_index}" 
            color_names.append(new_color)
        else:
            new_color = color_names[color_index]  

        # Asignar el color al nodo
        color_map[node] = new_color
        used_colors.add(new_color)

    return color_map, used_colors

def draw_colored_graph(G, coloring_result):
    """Dibuja el grafo con los nodos coloreados."""
    plt.figure(figsize=(12, 8))
    
    # Obtener la lista de colores asignados en formato válido para Matplotlib
    unique_colors = list(set(coloring_result.values()))  # Obtener colores únicos
    color_mapping = {color: idx for idx, color in enumerate(unique_colors)}  # Asignar un índice a cada color
    node_colors = [color_mapping[coloring_result[node]] for node in G.nodes()]  # Convertir a índices

    pos = nx.spring_layout(G)  # Disposición del grafo
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, edge_color="gray", cmap=plt.cm.get_cmap("tab10"))

    plt.title("Grafo coloreado")
    plt.show()

# Función de verificación de factibilidad
def es_coloreo_valido(G, coloring):
    for u, v in G.edges():
        if coloring[u] == coloring[v]:  # Restricción violada
            print(f"Error: Los nodos {u} y {v} tienen el mismo color ({coloring[u]})")
            return False
    return True

# Cargar el archivo de ejemplo
graph = load_col_file("C:/Users/anton/OneDrive/Documentos/6SEM/TSO/Graph Coloring Problem/DIMACS/david.col")

# Mostrar información del grafo
print("\n********** Información del Grafo **********\n")
print(graph, "\n")  # Imprime la cantidad de nodos y aristas (conexiones)
print("Nodos:", graph.nodes(), "\n")  # Imprime los nodos
print("Aristas:", graph.edges(), "\n")  # Imprime las conexiones

# Aplicar el algoritmo de coloreo
coloring_result, used_colors = greedy_coloring(graph)

# Mostrar los colores asignados a cada nodo
print("\n********** Colores asignados **********\n")
for node, color in coloring_result.items():
    print(f"Nodo {node}: {color}")

# Imprimir cantidad de colores usados y la lista de colores utilizados
print(f"\nSe utilizaron {len(used_colors)} colores diferentes.")
print("Colores utilizados:", sorted(used_colors))

# Verificar la factibilidad del coloreo
if es_coloreo_valido(graph, coloring_result):
    print("\nEl coloreo es válido.")
else:
    print("\nEl coloreo NO es válido.")


# Dibujar el grafo coloreado
draw_colored_graph(graph, coloring_result)
 # type: ignore