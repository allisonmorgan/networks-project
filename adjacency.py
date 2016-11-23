import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from importfaculty import faculty_graph
import numpy as np

def main():

  # Vertices are ordered by prestige in the dataset
  adj = nx.to_numpy_matrix(faculty_graph, dtype=int)

  #fig = plt.figure(figsize=(15,10))
  #ax = fig.add_subplot(111)
  #cax = plt.matshow(adj, cmap=plt.cm.Blues)
  #fig.colorbar(cax)

  #ax.set_ylabel("PhD Granting Institution")
  #ax.set_xlabel("Faculty Placement Institution")

  #plt.savefig("results/adjacency.png")
  #plt.clf()

  # Scale adjacency matrix by a vertex's outdegree.
  # Edges i -> j are from row_i -> col_j
  groups = np.linspace(0, 100, 11, dtype=int)
  grouped_by_row = []
  for i, row in enumerate(adj):
    in_edges = [] 
    for rank, edges in enumerate(row[0].tolist()[0]):
      for j in range(int(edges)):
        in_edges.append(rank)
    grouped_row, _ = np.histogram(in_edges, groups)
    grouped_by_row.append(grouped_row)

  grouped = [np.zeros(len(groups)-1) for i in range(len(groups)-1)]
  for i, row in enumerate(grouped_by_row):
    for j in range(len(groups)-1):
      if i <= groups[j+1]:
        for k, elem in enumerate(row):
          grouped[j][k] += elem
        break

  fig = plt.figure(figsize=(15,10))
  ax = fig.add_subplot(111)
  cax = ax.matshow(grouped, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  ax.set_xticklabels(groups)
  ax.set_yticklabels(groups)

  ax.set_ylabel("PhD Granting Institution")
  ax.set_xlabel("Faculty Placement Institution")

  plt.savefig("results/weighted_adjacency.png")

if __name__ == "__main__":
    main()