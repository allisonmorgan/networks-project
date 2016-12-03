
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

from importcompsci import school_metadata as meta_cs, faculty_graph as g_cs
from importhistory import school_metadata as meta_his, faculty_graph as g_his
from importbusiness import school_metadata as meta_busi, faculty_graph as g_busi

def main():
  fig, axarray = plt.subplots(1, 3, figsize=(12,5))
  #fig = plt.figure(figsize=(15,10))

  for l, (g, title) in enumerate([(g_busi, "Business"), (g_cs, "Computer Science"), (g_his, "History")]):
    # Vertices are ordered by prestige in the dataset
    adj = nx.to_numpy_matrix(g, dtype=int)

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

    #ax = fig.add_subplot(111)
    cax = axarray[l].matshow(grouped, cmap=cm.gray_r)

    axarray[l].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axarray[l].yaxis.set_major_locator(ticker.MultipleLocator(1))

    axarray[l].set_xticklabels(groups)
    axarray[l].set_yticklabels(groups)

    if l == 0:
      axarray[l].set_ylabel("PhD Granting Institution")
      axarray[l].set_xlabel("Faculty Placement Institution")

    axarray[l].set_title(title, y=1.15)

  #fig.colorbar(cax)

  plt.savefig("results/grouped_adjacency.png")

if __name__ == "__main__":
    main()