import networkx as nx
import matplotlib.pyplot as plt

def load_col_file(filename):
    G = nx.Graph()
    
    with open(filename, "r") as file:  # Open the file to read
        for line in file:  # Go through the connections in the file
            parts = line.split()  # Returns a list with the fragments of the string
            print(parts)
            
            if not parts or parts[0] == 'c':  # Ignore comments
                continue
                
            elif parts[0] == 'p':  # If parts is equal to 'p', read the number of nodes and edges
                num_nodes, num_edges = int(parts[2]), int(parts[3])
                G.add_nodes_from(range(1, num_nodes + 1))  # Add nodes

            elif parts[0] == 'e':  # If parts is equal to 'e', read the edges
                u, v = int(parts[1]), int(parts[2])
                G.add_edge(u, v)  # Add the edges (connections)

    return G

# List of predefined colors
color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]

def greedy_coloring(G):
    # Step 1: Sort nodes by degree in descending order
    nodes_sorted = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)

    # Dictionary to store the color of each node
    color_map = {}

    # Set to store used colors without repetition
    used_colors = set()

    print("\n********** Node order and their degrees **********\n")
    
    # Steps 2 and 3: Assign colors to nodes in the established order
    for node in nodes_sorted:
        degree = G.degree(node)  # Get the number of neighbors of the node
        print(f"Node {node}: {degree} neighbors")  # Print number of neighbors

        # Get colors of already colored neighbors
        neighbor_colors = {color_map[neighbor] for neighbor in G.neighbors(node) if neighbor in color_map}
        
        # Find the first available color from the list
        color_index = 0
        while color_index < len(color_names) and color_names[color_index] in neighbor_colors:
            color_index += 1

        # If predefined colors run out, generate new colors
        if color_index >= len(color_names):
            new_color = f"Color-{color_index}" 
            color_names.append(new_color)
        else:
            new_color = color_names[color_index]  

        # Assign the color to the node
        color_map[node] = new_color
        used_colors.add(new_color)

    return color_map, used_colors

def draw_colored_graph(G, coloring_result):
    """Draw the graph with colored nodes."""
    plt.figure(figsize=(12, 8))
    
    # Get the list of assigned colors in a valid format for Matplotlib
    unique_colors = list(set(coloring_result.values()))  # Get unique colors
    color_mapping = {color: idx for idx, color in enumerate(unique_colors)}  # Assign an index to each color
    node_colors = [color_mapping[coloring_result[node]] for node in G.nodes()]  # Convert to indices

    pos = nx.spring_layout(G)  # Graph layout
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, edge_color="gray", cmap=plt.cm.get_cmap("tab10"))

    plt.title("Colored Graph")
    plt.show()

# Feasibility check function
def is_valid_coloring(G, coloring):
    for u, v in G.edges():
        if coloring[u] == coloring[v]:  # Violated constraint
            print(f"Error: Nodes {u} and {v} have the same color ({coloring[u]})")
            return False
    return True

# Load the example file
graph = load_col_file("C:/Users/anton/OneDrive/Documentos/6SEM/TSO/Graph Coloring Problem/DIMACS/david.col")

# Display graph information
print("\n********** Graph Information **********\n")
print(graph, "\n")  # Print the number of nodes and edges (connections)
print("Nodes:", graph.nodes(), "\n")  # Print the nodes
print("Edges:", graph.edges(), "\n")  # Print the connections

# Apply the coloring algorithm
coloring_result, used_colors = greedy_coloring(graph)

# Display the colors assigned to each node
print("\n********** Assigned Colors **********\n")
for node, color in coloring_result.items():
    print(f"Node {node}: {color}")

# Print number of colors used and the list of used colors
print(f"\n{len(used_colors)} different colors were used.")
print("Used colors:", sorted(used_colors))

# Verify the feasibility of the coloring
if is_valid_coloring(graph, coloring_result):
    print("\nThe coloring is valid.")
else:
    print("\nThe coloring is NOT valid.")

# Draw the colored graph
draw_colored_graph(graph, coloring_result)