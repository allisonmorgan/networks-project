from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from epidemic import SI, SIR, SIS
from importcompsci import faculty_graph_weighted, school_metadata

def main():
    print("nodes: {0}".format(len(faculty_graph_weighted.nodes())))

    n_trials = 50
    results_length = []
    results_size = []
    ps = np.linspace(0, 0.5, 5)
    for p in ps:
        results_length_i = []
        results_size_i = []
        print("prob: {0}".format(p))

        for node in faculty_graph_weighted.nodes():
            print("node %: {0}".format(node / float(faculty_graph_weighted.number_of_nodes())))
            trials_length = []
            trials_size = []
            for i in range(n_trials):
                epi = SI(faculty_graph_weighted.copy(), p=p)
                epi.infect_node(node)
                epi.simulate()
                trials_length.append(epi.length)
                trials_size.append(epi.size)

            results_length_i.append((school_metadata[node]["pi"], np.average(trials_length)))
            results_size_i.append((school_metadata[node]["pi"], np.average(trials_size)))

        results_length.append(results_length_i)
        results_size.append(results_size_i)

    colors = iter(cm.rainbow(np.linspace(0, 1, len(results_length))))
    fig = plt.figure()
    ax = plt.gca()
    for i, data in enumerate(results_length):
        ax.scatter(*zip(*data), color = next(colors), label = 'p={0}'.format(ps[i]))
    
    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Length')
    plt.legend(loc=2)
    plt.ylim(0, 10)

    #giant = max(nx.connected_component_subgraphs(faculty_graph_weighted.to_undirected()), key=len)
    #plt.axhline(y=nx.diameter(giant), linewidth=1, color='black')
    plt.savefig('results/weighted0-length-{}-trials.png'.format(n_trials))
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
    plt.savefig('results/weighted0-size-{}-trials.png'.format(n_trials))

if __name__ == "__main__":
    main()