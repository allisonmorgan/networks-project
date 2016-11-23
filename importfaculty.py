import networkx
import numpy as np

# Import vertex attributes
with open("data/CS_vertexlist.txt") as f:
    lines = f.readlines()

# Create a map of node number to attributes
rows = [line.strip().split('\t') for i, line in enumerate(lines) if i > 0]
school_metadata = {}
vertex_names = {}
for row in rows:
    pi, USN2010, NRC95, region, institution = float(row[1]), row[2], row[3], row[4].strip(), row[5]
    try:
        USN2010 = float(USN2010)
    except:
        USN2010 = np.nan

    try:
        NRC95 = float(NRC95)
    except:
        NRC95 = np.nan

    vertex_names[row[5]] = int(row[0])
    attributes = {"pi": pi, "USN2010": USN2010, "NRC95": NRC95, "region": region, "institution": institution}
    school_metadata[int(row[0])] = attributes

# Import edge list
with open("data/CS_edgelist.txt") as f:
    lines = f.readlines()

# Create list of edges
edgelist = [line.strip().split("\t") for i, line in enumerate(lines) if i > 0]
edges = []
for edge in edgelist:
    # TODO: Doesn't store information on edge attributes.
    # Do we want this information (gender, rank)?
    if int(edge[0]) == 206 or int(edge[1]) == 206:
        continue
    edges.append((int(edge[0]), int(edge[1])))

# Create an empty directed, multi-edge graph and add all edges
faculty_graph = networkx.MultiDiGraph()
faculty_graph.add_edges_from(edges)

# Create an empty directed, weighted graph
faculty_graph_weighted = networkx.DiGraph()
# Draw an edge from every node to every other node with equal 
# weight. Allow self-loops
for u in faculty_graph.nodes():
    for v in faculty_graph.nodes():
        if not faculty_graph_weighted.has_edge(u, v):
            faculty_graph_weighted.add_edge(u, v, {'weight': 1})

# Increment the weight by the number of edges that exist
for u, v in faculty_graph.edges_iter():
    faculty_graph_weighted[u][v]['weight'] += 1

# Scale by the maximum number of edges?
#for u, v in faculty_graph_weighted.edges_iter():
#    faculty_graph_weighted[u][v]['weight'] = float(faculty_graph_weighted[u][v]['weight']) / float(len(edges))

