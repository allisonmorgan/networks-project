from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pickle

from importcompsci import school_metadata as meta_cs, faculty_graph as g_cs, faculty_graph_weighted as gw_cs
from importhistory import school_metadata as meta_his, faculty_graph as g_his, faculty_graph_weighted as gw_his
from importbusiness import school_metadata as meta_busi, faculty_graph as g_busi, faculty_graph_weighted as gw_busi

from utils import avg_geodesic_path_length_from


DIR_CS_SI = "cache/CS_SI.p"
DIR_HIS_SI = "cache/HIS_SI.p"
DIR_BUSI_SI = "cache/BUSI_SI.p"
DIR_CS_SIR = "cache/CS_SIR.p"
DIR_HIS_SIR = "cache/HIS_SIR.p"
DIR_BUSI_SIR = "cache/BUSI_SIR.p"
DIR_CS_SIS = "cache/CS_SIS.p"
DIR_HIS_SIS = "cache/HIS_SIS.p"
DIR_BUSI_SIS = "cache/BUSI_SIS.p"


def plot_name_of_dir(cache_dir):
    return cache_dir.replace("/", "_")[:-2]


def graph_of_dir(directory):
    if "CS" in directory:
        if "weighted" in directory:
            return gw_cs
        return g_cs
    if "HIS" in directory:
        if "weighted" in directory:
            return gw_his
        return g_his
    if "BUSI" in directory:
        if "weighted" in directory:
            return gw_busi
        return g_busi


def meta_of_dir(directory):
    if "CS" in directory:
        return meta_cs
    if "HIS" in directory:
        return meta_his
    if "BUSI" in directory:
        return meta_busi


def normalize(graph, node, length):
    return float(length) / \
        avg_geodesic_path_length_from(node, graph)


def bad_node_of_dir(cache_dir):
    if "CS" in cache_dir:
        return 206
    if "HIS" in cache_dir:
        return 145
    if "BUSI" in cache_dir:
        return 113


def plot_si_prestige(cache_dir):
    cache = pickle.load(open(cache_dir, 'rb'))
    meta = meta_of_dir(cache_dir)
    graph = graph_of_dir(cache_dir)
    results_length = defaultdict(list)
    results_size = defaultdict(list)
    for p in cache["length"].keys():
        for node, lengths in cache["length"][p].items():
            print(np.average(lengths))
            if node is bad_node_of_dir(cache_dir):
                continue
            result = (meta[node]["pi"], normalize(graph, node, np.average(lengths)))
            results_length[p].append(result)
    for p in cache["size"].keys():
        for node, sizes in cache["size"][p].items():
            if node is bad_node_of_dir(cache_dir):
                continue
            result = (meta[node]["pi"], np.average(sizes))
            results_size[p].append(result)

    length_of_results = len(cache["length"].keys())
    print(length_of_results)

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure()
    ax = plt.gca()
    for p, data in sorted(results_length.items(), key=lambda x: x[0]):
        ax.scatter(*zip(*data), color=next(colors), label='p={0:.2f}'.format(p))

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Normalized Epidemic Length')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6})
    plt.ylim(0, 10)

    plt.savefig('results/test/length-results-of-{}.png'.format(plot_name_of_dir(cache_dir)))
    plt.clf()

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure()
    ax = plt.gca()
    for p, data in sorted(results_size.items(), key=lambda x: x[0]):
        ax.scatter(*zip(*data), color=next(colors), label='p={0:.2f}'.format(p))

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Size')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6})
    plt.ylim(0, 1)
    plt.savefig('results/test/size-results-of-{}.png'.format(plot_name_of_dir(cache_dir)))

#def plot_sir_or_sis(cache_dir):
#    cache = pickle.load(open(cache_dir, 'rb'))
#    results_p = []; results_r = []; results_s = []; results_l = []
#    for r in rs:
#        for p in ps:
#            print (r, p)
#            samples_l = []
#            samples_s = []
#            for i in range(n_networks):
#                #print('-- NEW EPI --')
#                g = bi_planted_partition(1000, 8, 0)
#                epi = SIS(g, p=p, r=r)
#                #epi = SI(g, p=p)
#
#                epi.infect_random_node()
#                epi.simulate()
#
#                samples_l.append(epi.length)
#                samples_s.append(epi.size)
#
#            results_r.append(r)
#            results_p.append(p)
#            results_s.append(np.average(samples_s))
#            results_l.append(np.average(samples_l))
#
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    X, Y = results_r, results_p
#    Z = results_l
#    ax.scatter(X, Y, Z)
#
#    ax.set_xlabel('Recovery Probability')
#    ax.set_ylabel('Transmission Probability')
#    ax.set_zlabel('Epidemic Length')
#
#    plt.show()
#    # TODO: Figure out the right way to save these plots
#    plt.clf()
#
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    X, Y = results_r, results_p
#    Z = results_s
#    ax.scatter(X, Y, Z)
#
#    ax.set_xlabel('Recovery Probability')
#    ax.set_ylabel('Transmission Probability')
#    ax.set_zlabel('Epidemic Size')
#
#    plt.show()
#    # TODO: Figure out the right way to save these plots
#    plt.clf()

def main():
    plot_si_prestige(DIR_HIS_SI)


if __name__ == "__main__":
    main()
