from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from epidemic import SI, SIR, SIS
from importfaculty import faculty_graph, school_metadata


def main():
    print("nodes: ", len(faculty_graph.nodes()))

    n_trials = 50
    results_length = []
    results_size = []
    ps = np.linspace(0, 0.5, 5)
    for p in ps:
        results_length_i = []
        results_size_i = []
        print("prob: ", p)

        for node in faculty_graph.nodes():
            print("node %: ", node / float(faculty_graph.number_of_nodes()))
            trials_length = []
            trials_size = []
            for i in range(n_trials):
                epi = SI(faculty_graph.copy(), p=p)
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

    giant = max(nx.connected_component_subgraphs(faculty_graph.to_undirected()), key=len)
    plt.axhline(y=nx.diameter(giant), linewidth=1, color='black')
    plt.savefig('results/length-{}-trials.png'.format(n_trials))
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
    plt.savefig('results/size-{}-trials.png'.format(n_trials))
    
    # Start of incremental averaging (haven't throughly checked if correct).
    #trial_number = 1
    #results_length = defaultdict(int)
    #results_size = defaultdict(int)
    #p = 0.1
    #while True:
    #    print(trial_number)
    #    for node in school_metadata.keys():
    #        #print(node / float(faculty_graph.number_of_nodes()))
    #        epi = SI(faculty_graph.copy(), p=p)
    #        epi.infect_node(node)
    #        epi.simulate()
    #        prev_avg_length = results_length[school_metadata[node]["pi"]]
    #        prev_avg_size = results_size[school_metadata[node]["pi"]]
             # http://math.stackexchange.com/questions/106700/incremental-averageing
    #        results_length[school_metadata[node]["pi"]] = prev_avg_length + \
    #            (epi.length - prev_avg_length) / float(trial_number)
    #        results_size[school_metadata[node]["pi"]] = prev_avg_size + \
    #            (epi.size - prev_avg_size) / float(trial_number)

    #    if np.mod(trial_number, 5) == 0:
    #        ax = plt.gca()
    #        ax.scatter(*zip(*results_length.items()))
    #        plt.xlabel('University prestige (pi)')
    #        plt.ylabel('Epidemic length')
    #        plt.savefig('results/length-{}-trials-{}-p.png'.format(trial_number, p))
    #        plt.clf()

    #        ax = plt.gca()
    #        ax.scatter(*zip(*sorted(results_size.items(), key=lambda x: x[0])))
    #        plt.xlabel('University prestige (pi)')
    #        plt.ylabel('Epidemic size')
    #        plt.savefig('results/size-{}-trials-{}-p.png'.format(trial_number, p))
    #        plt.clf()

    #    trial_number += 1


if __name__ == "__main__":
    main()
