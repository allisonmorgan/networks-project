from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from epidemic import SI, SIR, SIS
from network import bi_planted_partition
from importfaculty import faculty_graph, school_metadata


def main():
    n_trials = 50
    results_length = []
    results_size = []
    p = 0.5
    for node in faculty_graph.nodes():
        print(node / float(faculty_graph.number_of_nodes()))
        trials_length = []
        trials_size = []
        for i in range(n_trials):
            epi = SI(faculty_graph.copy(), p=p)
            epi.infect_node(node)
            epi.simulate()
            trials_length.append(epi.length)
            trials_size.append(epi.size)
        results_length.append((school_metadata[node]["pi"], np.average(trials_length)))
        results_size.append((school_metadata[node]["pi"], np.average(trials_size)))

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(*zip(*results_length))
    plt.xlabel('University prestige (pi)')
    plt.ylabel('Epidemic length')
    plt.savefig('results/length-{}-trials-{}-p.png'.format(n_trials, p))
    plt.clf()

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(*zip(*results_size))
    plt.xlabel('University prestige (pi)')
    plt.ylabel('Epidemic size')
    plt.savefig('results/size-{}-trials-{}-p.png'.format(n_trials, p))


if __name__ == "__main__":
    main()
