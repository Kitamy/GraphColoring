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

    print("\n********** Node order and their degrees **********\n")
    
    for node in nodes_sorted:
        degree = G.degree(node)
        print(f"Node {node}: {degree} neighbors")

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

    plt.title("Colored Graph")

    # Show total execution time just before displaying the graph
    end_total = time.time()
    print(f"\nTotal execution time: {end_total - start_time:.4f} seconds")

    plt.show()

def is_coloring_valid(G, coloring):
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            print(f"Error: Nodes {u} and {v} have the same color ({coloring[u]})")
            return False
    return True

# Start measuring total time
start_total = time.time()

# Load the graph from the file. NOTE: Change the directory to wherever it is in the instance file.
graph = load_col_file("DIMACS/david.col")

# Display graph information
print("\n********** Graph Information **********\n")
print(graph, "\n")
print("Nodes:", graph.nodes(), "\n")
print("Edges:", graph.edges(), "\n")

# Run greedy coloring algorithm
coloring_result, used_colors = greedy_coloring(graph)

# Show coloring results
print("\n********** Assigned Colors **********\n")
for node, color in coloring_result.items():
    print(f"Node {node}: {color}")

print(f"\n{len(used_colors)} different colors were used.")
print("Used colors:", sorted(used_colors))

# Verify if the coloring is valid
if is_coloring_valid(graph, coloring_result):
    print("\nThe coloring is valid.")
else:
    print("\nThe coloring is NOT valid.")

# Draw the graph and show total time just before the graph
draw_colored_graph(graph, coloring_result, start_total)
