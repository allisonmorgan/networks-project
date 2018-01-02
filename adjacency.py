
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

from importcompsci import school_metadata as meta_cs, faculty_graph as g_cs
from importhistory import school_metadata as meta_his, faculty_graph as g_his
from importbusiness import school_metadata as meta_busi, faculty_graph as g_busi
import plot_utils

def main():
  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  #fig = plt.figure(figsize=(15,10))

  for l, (g, title) in enumerate([(g_cs, "Computer Science")]):
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
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    colors = iter(cm.rainbow(np.linspace(0, 1, 3)))
    #print(colors, next(colors))
    r,g,b = next(colors)[:3]  # Unpack RGB vals (0. to 1., not 0 to 255).
    cdict = {'red':   ((0.0,  1.0, 1.0),
                   (1.0,  r, r)),
         'green': ((0.0,  1.0, 1.0),
                   (1.0,  g, g)),
         'blue':  ((0.0,  1.0, 1.0),
                   (1.0,  b, b))}
    custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
    cax = ax.matshow(grouped, cmap=custom_cmap)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xticklabels(groups, fontsize=12)
    ax.set_yticklabels(groups, fontsize=12)

    if l == 0:
      ax.set_ylabel(r"Prestige of PhD Institution ($\pi$)", fontsize=16)
      ax.set_xlabel(r"Prestige of Hiring Institution ($\pi$)", fontsize=16)
      #ax.xaxis.set_label_coords(0.5, -.10)

    #ax.set_title(title, y=1.15, fontsize=16)
  
  plot_utils.finalize(ax)

  #fig.colorbar(cax)
  plt.tight_layout()
  plt.savefig("results/grouped_adjacency.eps", format='eps', dpi=1000) #bbox_inches='tight')

if __name__ == "__main__":
    main()