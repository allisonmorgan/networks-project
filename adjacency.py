import networkx as nx
import matplotlib.pyplot as plt
from importfaculty import faculty_graph
import numpy as np

def main():

  # Vertices are ordered by prestige in the dataset
  adj = nx.to_numpy_matrix(faculty_graph, dtype=float)

  # Scale adjacency matrix by a vertex's outdegree.
  # Edges i->j are from row_i -> col_j
  for i, row in enumerate(adj):
    scaled = []
    for j, cell in enumerate(row[0]):
      #print j
      scaled.append(cell/float(faculty_graph.in_degree(j+1)))
    
    adj[i] = scaled
    #print (i, adj[i])

  plt.matshow(adj, cmap=plt.cm.Blues)
  plt.show()