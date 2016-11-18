from importfaculty import faculty_graph, vertex_names, school_metadata
import matplotlib.pyplot as plt
import csv
import numpy as np

# Median h-index by school
h_b_insitution = {}
with open("data/gs_info.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for i, line in enumerate(tsvreader):
        # Skip the header
        if i == 0:
            continue

        if not h_b_insitution.has_key(line[1]):
            h_b_insitution[line[1]] = []
        h_b_insitution[line[1]].append(int(line[3]))

median_h_b_insitution = {}
for (institution, h_indices) in h_b_insitution.items():
    median_h_b_insitution[vertex_names[institution]] = np.median(h_indices)


"""
fig = plt.figure()
ax = plt.gca()

x = []; y = []
for (vertex, h_index) in median_h_b_insitution.items():
	x.append(school_metadata[vertex]['pi'])
	y.append(h_index)

ax.scatter(x, y)
    
plt.xlabel('University Prestige (pi)')
plt.ylabel('Median H-Index')
plt.savefig("results/h-index_v_rank.png")
plt.clf()
"""

"""
centrality = nx.closeness_centrality(faculty_graph)

fig = plt.figure()
ax = plt.gca()

x = []; y = []
for (vertex, c) in centrality.items():
    x.append(school_metadata[vertex]['pi'])
    y.append(c)

ax.scatter(x, y)
    
plt.xlabel('University Prestige (pi)')
plt.ylabel('Closeness Centrality')
plt.savefig("results/centrality_v_rank.png")
plt.clf()
"""
