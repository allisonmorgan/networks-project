from collections import defaultdict
from sklearn.linear_model import LinearRegression, LogisticRegression
import math
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


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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
    return float(length) / avg_geodesic_path_length_from(node, graph)


def bad_node_of_dir(cache_dir):
    if "CS" in cache_dir:
        return 206
    if "HIS" in cache_dir:
        return 145
    if "BUSI" in cache_dir:
        return 113

def plot_centrality():
    colors = iter(cm.rainbow(np.linspace(0, 1, 3)))
    fig = plt.figure()
    ax = plt.gca()

    for faculty_graph, school_metadata, dept in [(g_cs, meta_cs, "Computer Science"), (g_busi, meta_busi, "Business"), (g_his, meta_his, "History")]:
        centrality = nx.closeness_centrality(faculty_graph)

        x = []; y = []
        for (vertex, c) in centrality.items():
            x.append(school_metadata[vertex]['pi'])
            y.append(c)

        ax.scatter(x, y, label=dept, color=next(colors))
    
    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Closeness Centrality')
    plt.legend(loc='upper right', prop={'size': 9}, fontsize='large')
    plt.savefig("results/centrality.png")
    plt.clf()

# epidemic size and length versus prestige for various infection probabilities p
def plot_si_prestige(cache_dir):
    cache = pickle.load(open(cache_dir, 'rb'))
    meta = meta_of_dir(cache_dir)
    graph = graph_of_dir(cache_dir)
    results_length = defaultdict(list)
    results_size = defaultdict(list)
    for p in cache["length"].keys():
        for node, lengths in cache["length"][p].items():
            if node is bad_node_of_dir(cache_dir):
                continue

            avg = np.average(lengths)
            if not np.isnan(avg) and not np.isinf(avg):
                result = (meta[node]["pi"], normalize(graph, node, avg))
                results_length[p].append(result)

        results_length[p] = sorted(results_length[p], key=lambda x: x[0])

    for p in cache["size"].keys():
        for node, sizes in cache["size"][p].items():
            if node is bad_node_of_dir(cache_dir):
                continue

            avg = np.average(sizes)
            if not np.isnan(avg) and not np.isinf(avg):
                result = (meta[node]["pi"], avg)
                results_size[p].append(result)

        results_size[p] = sorted(results_size[p], key=lambda x: x[0])

    length_of_results = len(cache["length"].keys())
    #print(length_of_results)

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for p, data in sorted(results_length.items(), key=lambda x: x[0]):
        c = next(colors)
        ax.scatter(*zip(*data), color=c, label='p = {0:.2f}'.format(p))
        #ax.plot(*zip(*data), color=next(colors), label='p = {0:.2f}'.format(p), marker = 'o')
        
        # fit a linear curve to this
        x = np.array([pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])
        y = np.array([length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])

        regr = LinearRegression()
        regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        interval = np.array([min(x), max(x)])
        ax.plot(interval, interval*regr.coef_[0] + regr.intercept_, color=c)

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Normalized Epidemic Length')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9}, fontsize='large')
    plt.ylim(0, 10)

    plt.savefig('results/test/length-results-of-{}.png'.format(plot_name_of_dir(cache_dir)))
    plt.clf()

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for p, data in sorted(results_size.items(), key=lambda x: x[0]):
        c = next(colors)
        ax.scatter(*zip(*data), color=c, label='p = {0:.2f}'.format(p))
        #ax.plot(*zip(*data), color=next(colors), label='p = {0:.2f}'.format(p), marker = 'o')

        #if p > 0:
            # fit a logistic curve to this
            #x = np.array([pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])
            #y = np.array([str(length) for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])

            #regr = LogisticRegression()
            #regr.fit(x.reshape(-1, 1), y)

            #loss = sigmoid(x * regr.coef_ + regr.intercept_).ravel()
            #ax.plot(x, loss, "-", color=c)

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Size')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9}, fontsize='large')
    plt.ylim(0, 1)
    plt.savefig('results/test/size-results-of-{}.png'.format(plot_name_of_dir(cache_dir)))

# epidemic size and length versus prestige for various spreading parameters p/r
def plot_sis_or_sir_prestige(cache_dir):
    cache = pickle.load(open(cache_dir, 'rb'))
    meta = meta_of_dir(cache_dir)
    graph = graph_of_dir(cache_dir)
    results_length = defaultdict(list)
    results_size = defaultdict(list)
    for (p, r) in cache["length"].keys():
        if r == 0.0:
            continue # can't divide by zero below. is this the right thing to do?
        for node, lengths in cache["length"][p, r].items():
            #print(np.average(lengths))
            if node is bad_node_of_dir(cache_dir):
                continue
            result = (meta[node]["pi"], normalize(graph, node, np.average(lengths)))
            results_length[p/r].append(result)
    for (p, r) in cache["size"].keys():
        if r == 0.0:
            continue
        for node, sizes in cache["size"][p, r].items():
            if node is bad_node_of_dir(cache_dir):
                continue
            result = (meta[node]["pi"], np.average(sizes))
            results_size[p/r].append(result)

    # different values of p and r will return the same p/r. average these values?
    for ratio, data in results_length.copy().items():
        avg_by_prestige = defaultdict(list)
        for pi, length in data:
            avg_by_prestige[pi].append(length)

        results_length[ratio] = [(pi, np.average(lengths)) for pi, lengths in avg_by_prestige.items()]
        results_length[ratio] = sorted(results_length[ratio], key=lambda x: x[0])
    #print(results_length[1])
    for ratio, data in results_size.copy().items():
        avg_by_prestige = defaultdict(list)
        for pi, size in data:
            avg_by_prestige[pi].append(size)

        results_size[ratio] = [(pi, np.average(sizes)) for pi, sizes in avg_by_prestige.items()]
        results_size[ratio] = sorted(results_size[ratio], key=lambda x: x[0])

    length_of_results = len(results_size.keys())
    #print(length_of_results)

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for ratio, data in sorted(results_length.items(), key=lambda x: x[0]):
        ax.scatter(*zip(*data), color=next(colors), label='p/r = {0:.2f}'.format(ratio))
        #ax.plot(*zip(*data), color=next(colors), label='p/r = {0:.2f}'.format(ratio), marker = 'o')

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Normalized Epidemic Length')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9}, fontsize='large')
    plt.ylim(0, 10)

    plt.savefig('results/test/length-results-of-{}.png'.format(plot_name_of_dir(cache_dir)))
    plt.clf()

    colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for ratio, data in sorted(results_size.items(), key=lambda x: x[0]):
        ax.scatter(*zip(*data), color=next(colors), label='p/r = {0:.2f}'.format(ratio))
        #ax.plot(*zip(*data), color=next(colors), label='p/r = {0:.2f}'.format(ratio), marker = 'o')

    plt.xlabel('University Prestige (pi)')
    plt.ylabel('Epidemic Size')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9}, fontsize='large')
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
    #plot_si_prestige(DIR_CS_SI)
    #plot_sis_or_sir_prestige(DIR_CS_SIR)
    plot_centrality()

if __name__ == "__main__":
    main()
