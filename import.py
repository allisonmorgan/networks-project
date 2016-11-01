import networkx
import numpy as np

# Import vertex attributes
with open("data/CS_vertexlist.txt") as f:
    lines = f.readlines()

# Create a map of node number to attributes
rows = [line.strip().split('\t') for i, line in enumerate(lines) if i > 0]
school_metadata = {}
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
    edges.append((int(edge[0]), int(edge[1])))

# Create an empty directed graph and add all edges
# TODO: This collapses all edges into a single edge.
# Do we want to add edge weight here?
g = networkx.DiGraph()
g.add_edges_from(edges)
