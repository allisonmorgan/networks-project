from collections import defaultdict
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
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

DIR_CS_SI_JUMP_PROBABILITY = "cache/random_hops/jump_probability/CS_SI.p"
DIR_HIS_SI_JUMP_PROBABILITY = "cache/random_hops/jump_probability/HIS_SI.p"
DIR_BUSI_SI_JUMP_PROBABILITY = "cache/random_hops/jump_probability/BUSI_SI.p"
DIR_CS_SIR_JUMP_PROBABILITY = "cache/random_hops/jump_probability/CS_SIR.p"
DIR_HIS_SIR_JUMP_PROBABILITY = "cache/random_hops/jump_probability/HIS_SIR.p"
DIR_BUSI_SIR_JUMP_PROBABILITY = "cache/random_hops/jump_probability/BUSI_SIR.p"
DIR_CS_SIS_JUMP_PROBABILITY = "cache/random_hops/jump_probability/CS_SIS.p"
DIR_HIS_SIS_JUMP_PROBABILITY = "cache/random_hops/jump_probability/HIS_SIS.p"
DIR_BUSI_SIS_JUMP_PROBABILITY = "cache/random_hops/jump_probability/BUSI_SIS.p"

dirs = [DIR_CS_SI, DIR_HIS_SI, DIR_BUSI_SI, DIR_CS_SIR, DIR_HIS_SIR, DIR_BUSI_SIR, DIR_CS_SIS, DIR_HIS_SIS, DIR_BUSI_SIS]

all_departments_SI = [("Business", DIR_BUSI_SI), ("Computer Science", DIR_CS_SI), ("History", DIR_HIS_SI)]
all_departments_SIR = [("Business", DIR_BUSI_SIR), ("Computer Science", DIR_CS_SIR), ("History", DIR_HIS_SIR)]
all_departments_SIS = [("Business", DIR_BUSI_SIS), ("Computer Science", DIR_CS_SIS), ("History", DIR_HIS_SIS)]

all_departments_SI_random_jump = [("Business", DIR_BUSI_SI_JUMP_PROBABILITY), ("Computer Science", DIR_CS_SI_JUMP_PROBABILITY), ("History", DIR_HIS_SI_JUMP_PROBABILITY)]
all_departments_SIR_random_jump = [("Business", DIR_BUSI_SIR_JUMP_PROBABILITY), ("Computer Science", DIR_CS_SIR_JUMP_PROBABILITY), ("History", DIR_HIS_SIR_JUMP_PROBABILITY)]
all_departments_SIS_random_jump = [("Business", DIR_BUSI_SIS_JUMP_PROBABILITY), ("Computer Science", DIR_CS_SIS_JUMP_PROBABILITY), ("History", DIR_HIS_SIS_JUMP_PROBABILITY)]

def curve(x, h, a, k):
    return h / (1 + np.exp(a * (x - k)))

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

def average_across_infection_probability(tuples):
    dictionary = defaultdict(list)
    for (infection_prob, size) in tuples:
        dictionary[infection_prob].append(size)

    return [(infection_prob, np.average(sizes)) for infection_prob, sizes in dictionary.items()]


def bad_node_of_dir(cache_dir):
    if "CS" in cache_dir:
        return 206
    if "HIS" in cache_dir:
        return 145
    if "BUSI" in cache_dir:
        return 113


def n_trials_of_dir(cache_dir):
    cache = pickle.load(open(cache_dir, 'rb'))
    try:
        a_prob_key = cache["size"].iterkeys().next()
        a_node_key = cache["size"][a_prob_key].iterkeys().next()
        a_sizes_val = cache["size"][a_prob_key][a_node_key]
        return len(a_sizes_val)
    except:
        return 0

def print_n_trials():
    for a_dir in dirs:
        print(a_dir)
        print(n_trials_of_dir(a_dir))

def plot_centrality():
    colors = iter(cm.rainbow(np.linspace(0, 1, 3)))
    markers = Line2D.filled_markers
    fig = plt.figure(figsize=(6.9,5.))
    ax = plt.gca()

    for i, (faculty_graph, school_metadata, dept) in enumerate([(g_cs, meta_cs, "Computer Science"), (g_busi, meta_busi, "Business"), (g_his, meta_his, "History")]):
        centrality = nx.closeness_centrality(faculty_graph)

        x = []; y = []
        max_pi = 0
        for (vertex, c) in centrality.items():
            x.append(school_metadata[vertex]['pi'])
            y.append(c)
            if school_metadata[vertex]['pi'] > max_pi:
                max_pi = school_metadata[vertex]['pi']

        ax.scatter(x, y, label=dept, color=next(colors), marker=markers[i])

    plt.xlabel(r'University Prestige $\pi$', fontsize=16)
    plt.ylabel(r'Closeness Centrality', fontsize=16)
    plt.xlim(0, max_pi)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right', prop={'size': 12}, fontsize='large')
    plt.savefig("results/centrality.png")
    plt.clf()


# epidemic size versus prestige for various infection probabilities p
def plot_si_prestige_size(cache_dirs):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for i, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[i]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_size = defaultdict(list)
        for p in cache["size"].keys():
            for node, sizes in cache["size"][p].items():
                if node is bad_node_of_dir(cache_dir):
                    continue

                avg = np.average(sizes)
                if not np.isnan(avg) and not np.isinf(avg):
                    result = (meta[node]["pi"], avg)
                    results_size[p].append(result)

            results_size[p] = sorted(results_size[p], key=lambda x: x[0])

        filtered = sorted(cache["size"].keys())[1::2]
        length_of_results = len(filtered)

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        markers = Line2D.filled_markers; count = -1
        for p, data in sorted(results_size.items(), key=lambda x: x[0]):
            if p not in filtered:
                continue
            c = next(colors); count += 1; m = markers[count]
            ax.scatter(*zip(*data), color=c, label='{0:.2f}'.format(p), s=10, marker=m)

            x = [pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]
            max_pi = max(x)
            if p > 0:
                # fit a logistic curve to this
                y = [length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]

                popt, pcov = curve_fit(curve, np.array(x), np.array(y), bounds=(0., [1., 2., 200.]))
                print("infection probability: {0}\tcurve_fit: {1}".format(p, popt))
                y = curve(x, *popt)

                ax.plot(x, y, color=c)

        ax.set_xlim(0, max_pi)
        ax.set_title(title, y=1.05, fontsize=16)
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.set_xlabel(r'University Prestige $\pi$', fontsize=16)
            ax.set_ylabel(r'Epidemic Size', fontsize=16)

    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}, fontsize='large', title=r'$p$')
    plt.savefig('results/test/size-results-of-ALL-SI.png')

def plot_si_prestige_length(cache_dirs, ylim=(0,5)):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for i, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[i]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_length = defaultdict(list)
        for p in cache["length"].keys():
            for node, lengths in cache["length"][p].items():
                if node is bad_node_of_dir(cache_dir):
                    continue

                avg = np.average(lengths)
                if not np.isnan(avg) and not np.isinf(avg):
                    result = (meta[node]["pi"], normalize(graph, node, avg))
                    results_length[p].append(result)

            results_length[p] = sorted(results_length[p], key=lambda x: x[0])

        for ratio, data in results_length.copy().items():
            avg_by_prestige = defaultdict(list)
            for pi, length in data:
                avg_by_prestige[pi].append(length)

            results_length[ratio] = [(pi, np.average(lengths)) for pi, lengths in avg_by_prestige.items()]
            results_length[ratio] = sorted(results_length[ratio], key=lambda x: x[0])

        filtered = sorted(cache["length"].keys())[1::2]
        length_of_results = len(filtered)

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        markers = Line2D.filled_markers; count = -1
        for p, data in sorted(results_length.items(), key=lambda x: x[0]):
            if p not in filtered:
                continue
            c = next(colors); count += 1; m = markers[count]
            ax.scatter(*zip(*data), color=c, label='{0:.2f}'.format(p), s=10, marker=m)

            # fit a linear curve to this
            x = np.array([pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])
            max_pi = max(x)
            y = np.array([length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])

            regr = LinearRegression()
            regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
            interval = np.array([min(x), max(x)])
            print("infection probability: {0}\tcurve_fit: {1}".format(p, [regr.coef_[0], regr.intercept_]))
            ax.plot(interval, interval*regr.coef_[0] + regr.intercept_, color=c)

        ax.set_xlim(0, max_pi)
        ax.set_title(title, y=1.05, fontsize=16)
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.set_xlabel(r'University Prestige $\pi$', fontsize=16)
            ax.set_ylabel(r'Normalized Epidemic Length', fontsize=16)

    plt.ylim(ylim)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}, fontsize='large', title=r'$p$')
    plt.savefig('results/test/length-results-of-ALL-SI.png')

def plot_sis_or_sir_prestige_size(cache_dirs, epidemic_type, ylim=(0,1)):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for i, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[i]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_size = defaultdict(list)
        for (p, r) in cache["size"].keys():
            if r == 0.0:
                continue # can't divide by zero below. is this the right thing to do?
            for node, sizes in cache["size"][p, r].items():
                if node is bad_node_of_dir(cache_dir):
                    continue
                result = (meta[node]["pi"], np.average(sizes))
                results_size[p/r].append(result)

        # different values of p and r will return the same p/r. average these values?
        for ratio, data in results_size.copy().items():
            avg_by_prestige = defaultdict(list)
            for pi, size in data:
                avg_by_prestige[pi].append(size)

            results_size[ratio] = [(pi, np.average(sizes)) for pi, sizes in avg_by_prestige.items()]
            results_size[ratio] = sorted(results_size[ratio], key=lambda x: x[0])

        filtered = ["1.0", "2.0", "3.0", "4.0", "5.0"]
        length_of_results = len(filtered)

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        markers = Line2D.filled_markers; count = -1
        for ratio, data in sorted(results_size.items(), key=lambda x: x[0]):
            if "%.1f" % ratio not in filtered:
                continue
            c = next(colors); count += 1; m = markers[count]
            ax.scatter(*zip(*data), color=c, label='{0:.2f}'.format(ratio), marker=m)

            x = [pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]
            max_pi = max(x)
            if ratio > 0:
                # fit a logistic curve to this
                y = [length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]

                popt, pcov = curve_fit(curve, np.array(x), np.array(y), bounds=(0., [1., 2., 200.]))
                print("infection probability: {0}\tcurve_fit: {1}".format(ratio, popt))
                y = curve(x, *popt)

                ax.plot(x, y, color=c)
            #ax.plot(*zip(*data), color=next(colors), label='p/r = {0:.2f}'.format(ratio), marker = 'o')

        ax.set_xlim(0, max_pi)
        ax.set_title(title, y=1.05, fontsize=16)
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.set_xlabel(r'University Prestige $\pi$', fontsize=16)
            ax.set_ylabel(r'Epidemic Size', fontsize=16)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}, fontsize='large', title=r'$p/r$')
    plt.ylim(ylim)
    plt.savefig('results/test/size-results-of-ALL-{}.png'.format(epidemic_type))
    plt.clf()

def plot_sis_or_sir_prestige_length(cache_dirs, epidemic_type, ylim=(0,10)):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for i, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[i]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_length = defaultdict(list)
        for (p, r) in cache["length"].keys():
            if r == 0.0:
                continue # can't divide by zero below. is this the right thing to do?
            for node, lengths in cache["length"][p, r].items():
                #print(np.average(lengths))
                if node is bad_node_of_dir(cache_dir):
                    continue
                result = (meta[node]["pi"], normalize(graph, node, np.average(lengths)))
                results_length[p/r].append(result)

        # different values of p and r will return the same p/r. average these values?
        for ratio, data in results_length.copy().items():
            avg_by_prestige = defaultdict(list)
            for pi, length in data:
                avg_by_prestige[pi].append(length)

            results_length[ratio] = [(pi, np.average(lengths)) for pi, lengths in avg_by_prestige.items()]
            results_length[ratio] = sorted(results_length[ratio], key=lambda x: x[0])

        filtered = ["1.0", "2.0", "3.0", "4.0", "5.0"]
        length_of_results = len(filtered)

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        markers = Line2D.filled_markers; count = -1
        for ratio, data in sorted(results_length.items(), key=lambda x: x[0]):
            if "%.1f" % ratio not in filtered:
                continue
            c = next(colors); count += 1
            ax.scatter(*zip(*data), color=c, label='{0:.2f}'.format(ratio), marker=markers[count])

            # fit a linear curve to this
            x = np.array([pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])
            max_pi = max(x)
            y = np.array([length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)])

            regr = LinearRegression()
            regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
            interval = np.array([min(x), max(x)])
            print("infection probability: {0}\tcurve_fit: {1}".format(ratio, [regr.coef_[0], regr.intercept_]))
            ax.plot(interval, interval*regr.coef_[0] + regr.intercept_, color=c)

        ax.set_xlim(0, max_pi)
        ax.set_title(title, y=1.05, fontsize=16)
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.set_xlabel(r'University Prestige $\pi$', fontsize=16)
            ax.set_ylabel(r'Normalized Epidemic Length', fontsize=16)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}, fontsize='large', title=r'$p/r$')
    plt.ylim(ylim)
    plt.savefig('results/test/length-results-of-ALL-{0}.png'.format(epidemic_type))
    plt.clf()


def plot_random_hop_size(cache_dirs, epidemic_type, ylim=(0, 1)):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for i, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[i]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_size = defaultdict(list)
        for p in cache["size"].keys():
            for node, sizes in cache["size"][p].items():
                if node is bad_node_of_dir(cache_dir):
                    continue

                avg = np.average(sizes)
                if not np.isnan(avg) and not np.isinf(avg):
                    result = (meta[node]["pi"], avg)
                    results_size[p].append(result)

            results_size[p] = sorted(results_size[p], key=lambda x: x[0])

        filtered = sorted(cache["size"].keys())[1::2]
        length_of_results = len(filtered)

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        markers = Line2D.filled_markers; count = -1
        for p, data in sorted(results_size.items(), key=lambda x: x[0]):
            if p not in filtered:
                continue
            c = next(colors); count += 1; m = markers[count]
            ax.scatter(*zip(*data), color=c, label='{0:.2f}'.format(p), s=10, marker=m)

            x = [pi for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]
            max_pi = max(x)
            if p > 0:
                # fit a logistic curve to this
                y = [length for (pi, length) in data if not np.isnan(length) and not np.isinf(length)]

                popt, pcov = curve_fit(curve, np.array(x), np.array(y), bounds=(0., [1., 2., 200.]))
                y = curve(x, *popt)

                ax.plot(x, y, color=c)

        ax.set_xlim(0, max_pi)
        ax.set_title(title, y=1.05, fontsize=16)
        if i == 0: 
            ax.set_xlabel(r'University Prestige $\pi$', fontsize=16)
            ax.set_ylabel(r'Epidemic Size', fontsize=16)
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9}, fontsize='large', title=r"$j$")
    plt.ylim(ylim)
    plt.savefig('results/test/size-results-of-ALL-{}-random-hops.png'.format(epidemic_type))

# epidemic size versus infection probability for all institutions
def plot_size_infection_probability(cache_dirs, threshold=0.00, bins=range(0, 100, 10), average=False):
    fig, axarray = plt.subplots(1, len(cache_dirs), figsize=(6.9*2, 5.0), sharey=True)
    for j, ax in enumerate(axarray):
        (title, cache_dir) = cache_dirs[j]
        print("title: {0}".format(title))
        cache = pickle.load(open(cache_dir, 'rb'))
        meta = meta_of_dir(cache_dir)
        graph = graph_of_dir(cache_dir)
        results_size = defaultdict(list)

        for p in cache["size"].keys():
            for node, sizes in cache["size"][p].items():
                if node is bad_node_of_dir(cache_dir):
                    continue

                pi = meta[node]["pi"]
                avg = np.average(sizes)
                if not np.isnan(avg) and not np.isinf(avg):
                    result = (p, avg)
                    results_size[pi].append(result)

        # remove data below a threshold
        for pi, data in results_size.copy().items():
            trend = [size for _, size in data]
            if max(trend) <= threshold:
                del results_size[pi]
            else:
                results_size[pi] = sorted(data, key=lambda x: x[0])

        # bin the remaining data
        if bins != None:
            left_endpoint = bins[0]
            percentiles = np.percentile(results_size.keys(), bins[1:])
            bin_means = defaultdict(list)
            for i, bin_edge in enumerate(percentiles):
                bin_values = []
                for pi in results_size.keys():
                    if left_endpoint < pi <= bin_edge:
                        bin_values.extend(results_size[pi])
            
                bin_means[(i+1)] = average_across_infection_probability(bin_values)
                left_endpoint = bin_edge
            results_size = bin_means

        # find average trend over all prestige levels
        if average:
            averages = defaultdict(list)
            for pi, data in results_size.items():
                for x, y in data:
                    averages[x].append(y)

            results_size = defaultdict(list)
            for p, sizes in averages.items():
                results_size[1].append((p, np.average(sizes)))

        length_of_results = len(results_size.keys())

        colors = iter(cm.rainbow(np.linspace(0, 1, length_of_results)))
        for pi, data in sorted(results_size.items(), key=lambda x: x[0]):
            data = sorted(data, key=lambda x: x[0])
            c = next(colors)
            ax.scatter(*zip(*data), color=c, label='{0:f}'.format(pi*10))


            # fit a logistic curve to this
            if not average:
                x = [p for (p, size) in data if not np.isnan(size) and not np.isinf(size)]
                y = [size for (p, size) in data if not np.isnan(size) and not np.isinf(size)]

                popt, pcov = curve_fit(curve, np.array(x), np.array(y), bounds=([0., -150., -5.], [1., 0., 5.]))
                y = curve(x, *popt)
                print("prestige: {0}\tcurve_fit: {1}".format(pi, popt))
                ax.plot(x, y, color=c)


        ax.set_title(title, y=1.05, fontsize=16)
        ax.tick_params(labelsize=12)
        if j == 0:
            plt.ylim(0, 1.)
            ax.set_xlabel(r'Infection Probability', fontsize=16)
            ax.set_ylabel(r'Epidemic Size', fontsize=16)

    if bins != None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}, fontsize='large', title=r'$\pi$')
    plt.savefig('results/test/infectious/size-results-of-ALL-SI.png')

# TODO: 3D plots
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

    #plot_centrality()

    #plot_si_prestige_size(all_departments_SI)
    #plot_si_prestige_length(all_departments_SI, ylim=(0,5))

    #plot_sis_or_sir_prestige_length(all_departments_SIR, "SIR", ylim=(0,2.5))
    #plot_sis_or_sir_prestige_size(all_departments_SIR, "SIR", ylim=(0, 0.6))

    #plot_sis_or_sir_prestige_length(all_departments_SIS, "SIS", ylim=(0, 100)) 
    #plot_sis_or_sir_prestige_size(all_departments_SIS, "SIS", ylim=(0,0.1))

    #plot_random_hop_size(all_departments_SIS_random_jump, "SIS", ylim=(0,0.15))
    #plot_random_hop_size(all_departments_SIR_random_jump, "SIR", ylim=(0,0.5))
    #plot_random_hop_size(all_departments_SI_random_jump, "SI", ylim=(0,1))
    #plot_size_infection_probability(all_departments_SI)
    
    #for (title, cache_dir) in all_departments_SI_random_jump:
    #    print("title: {0}\tnumber of SI trials: {1}".format(title, n_trials_of_dir(cache_dir)))

    #for (title, cache_dir) in all_departments_SIR_random_jump:
    #    print("title: {0}\tnumber of SIR trials: {1}".format(title, n_trials_of_dir(cache_dir)))

    #for (title, cache_dir) in all_departments_SIS_random_jump:
    #    print("title: {0}\tnumber of SIS trials: {1}".format(title, n_trials_of_dir(cache_dir)))


if __name__ == "__main__":
    main()
