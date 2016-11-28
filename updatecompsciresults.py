from __future__ import print_function
from collections import defaultdict
from itertools import product
import pickle

import numpy as np

import networkx as nx

from epidemic import SI, SIR, SIS
from importcompsci import faculty_graph, school_metadata


def run_trials(si_trials=2, sir_trials=2, sis_trials=2):
    ps = np.linspace(0, 1, 11)
    rs = np.linspace(0, 1, 5, endpoint=False)

    #results = {"size": {}, "length": {}}
    #for p in ps:
    #    results["size"][p] = defaultdict(list)
    #    results["length"][p] = defaultdict(list)
    results = pickle.load(open("cache/CS_SI.p", "rb"))
    for trial in xrange(si_trials):
        print("Trial progress: {}".format(trial / float(si_trials)))
        for p in ps:
            print(p)
            for node in school_metadata.keys():
                epi = SI(faculty_graph.copy(), p=p)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/CS_SI.p", 'wb'))
    results.clear()
    print("SI done")

    #results = {"size": {}, "length": {}}
    #for p, r in product(ps, rs):
    #    results["size"][p, r] = defaultdict(list)
    #    results["length"][p, r] = defaultdict(list)
    results = pickle.load(open("cache/CS_SIR.p", "rb"))
    for trial in xrange(sir_trials):
        print("Trial progress: {}".format(trial / float(sir_trials)))
        for p, r in product(ps, rs):
            print((p,r))
            for node in school_metadata.keys():
                epi = SIR(faculty_graph.copy(), p=p, r=r)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p, r][node].append(epi.size)
                results["length"][p, r][node].append(epi.length)
    pickle.dump(results, open("cache/CS_SIR.p", 'wb'))
    results.clear()
    print("SIR done")

    #results = {"size": {}, "length": {}}
    #for p, r in product(ps, rs):
    #    results["size"][p, r] = defaultdict(list)
    #    results["length"][p, r] = defaultdict(list)
    results = pickle.load(open("cache/CS_SIS.p", "rb"))
    for trial in xrange(sis_trials):
        print("Trial progress: {}".format(trial / float(sis_trials)))
        for p, r in product(ps, rs):
            print((p,r))
            for node in school_metadata.keys():
                epi = SIS(faculty_graph.copy(), p=p, r=r)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p, r][node].append(epi.size)
                results["length"][p, r][node].append(epi.length)
    pickle.dump(results, open("cache/CS_SIS.p", 'wb'))
    results.clear()
    print("SIS done")


def main():
    # set number of for each trials here (0 or more)
    run_trials(si_trials=1, sir_trials=1, sis_trials=1)


if __name__ == "__main__":
    main()
