from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from epidemic import SI, SIR, SIS
from importfaculty import faculty_graph, school_metadata, faculty_graph_weighted


def main():
    g = faculty_graph_weighted
    print("nodes: ", len(g.nodes()))

    n_trials = 50
    results_length = []
    results_size = []
    ps = [0.001, 0.005, 0.01]
    for p in ps:
        results_length_i = []
        results_size_i = []
        print("prob: ", p)

        for node in g.nodes():
            print("node %: ", node / float(g.number_of_nodes()))
            trials_length = []
            trials_size = []
            for i in range(n_trials):
                epi = SI(g.copy(), p=p)
                epi.infect_node(node)
                epi.simulate()
                trials_length.append(epi.length)
                trials_size.append(epi.size)

            results_length_i.append((school_metadata[node]["pi"], np.average(trials_length)))
            results_size_i.append((school_metadata[node]["pi"], np.average(trials_size)))

        results_length.append(results_length_i)
        results_size.append(results_size_i)

    colors = iter(cm.Blues)
    fig = plt.figure()
    ax = plt.gca()
    for i, data in enumerate(results_length):
        ax.scatter(*zip(*data), color = next(colors), label = 'p={0}'.format(ps[i]))
    
    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Length')
    plt.legend(loc=2)
    plt.ylim(0, 10)

    #giant = max(nx.connected_component_subgraphs(g.to_undirected()), key=len)
    #plt.axhline(y=nx.diameter(giant), linewidth=1, color='black')
    plt.savefig('results/length-weighted-{}-trials.png'.format(n_trials))
    plt.clf()

    colors = iter(cm.rainbow(np.linspace(0, 1, len(results_length))))
    fig = plt.figure()
    ax = plt.gca()
    for i, data in enumerate(results_size):
        ax.scatter(*zip(*data), color = next(colors), label = 'p={0}'.format(ps[i]))

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Size')
    plt.legend(loc=2)
    plt.ylim(0, 1)
    plt.savefig('results/size-weighted-{}-trials.png'.format(n_trials))


if __name__ == "__main__":
    main()
