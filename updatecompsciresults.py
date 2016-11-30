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

def run_trials_graph_with_random_hops(si_trials=2, sir_trials=2, sis_trials=2):
    pjumps = np.linspace(0, 1, 5)
    fjumps = np.linspace(0, 1, 5)

    # vary jump probability
    results = {"size": {}, "length": {}}
    for p in pjumps:
        results["size"][p] = defaultdict(list)
        results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/jump_probability/CS_SI.p", "rb"))
    for trial in xrange(si_trials):
        print("Trial progress: {}".format(trial / float(si_trials)))
        for p in pjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SI(faculty_graph.copy(), p=0.1, random_jump_p=p, is_random_jump=True)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/jump_probability/CS_SI.p", 'wb'))
    results.clear()

    # vary fraction to jump to
    results = {"size": {}, "length": {}}
    for p in fjumps:
    results["size"][p] = defaultdict(list)
    results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/number_of_jumps/CS_SI.p", "rb"))
    for trial in xrange(si_trials):
        print("Trial progress: {}".format(trial / float(si_trials)))
        for p in fjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SI(faculty_graph.copy(), p=0.1, is_random_jump=True, n_random_jumps=p, random_jump_p=0.001)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/number_of_jumps/CS_SI.p", 'wb'))
    results.clear()

    print("SI done")

    results = {"size": {}, "length": {}}
    for p in pjumps:
        results["size"][p] = defaultdict(list)
        results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/jump_probability/CS_SIR.p", "rb"))
    for trial in xrange(sir_trials):
        print("Trial progress: {}".format(trial / float(sir_trials)))
        for p in pjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SIR(faculty_graph.copy(), p=0.1, r=0.2, random_jump_p=p, is_random_jump=True)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/jump_probability/CS_SIR.p", 'wb'))
    results.clear()

    results = {"size": {}, "length": {}}
    for p in fjumps:
        results["size"][p] = defaultdict(list)
        results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/number_of_jumps/CS_SIR.p", "rb"))
    for trial in xrange(sir_trials):
        print("Trial progress: {}".format(trial / float(sir_trials)))
        for p in fjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SIR(faculty_graph.copy(), p=0.1, r=0.2, is_random_jump=True, n_random_jumps=p, random_jump_p=0.001)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/number_of_jumps/CS_SIR.p", 'wb'))
    results.clear()

    print("SIR done")

    results = {"size": {}, "length": {}}
    for p in pjumps:
        results["size"][p] = defaultdict(list)
        results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/jump_probability/CS_SIS.p", "rb"))
    for trial in xrange(sir_trials):
        print("Trial progress: {}".format(trial / float(sir_trials)))
        for p in pjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SIS(faculty_graph.copy(), p=0.1, r=0.2, random_jump_p=p, is_random_jump=True)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/jump_probability/CS_SIS.p", 'wb'))
    results.clear()

    results = {"size": {}, "length": {}}
    for p in fjumps:
        results["size"][p] = defaultdict(list)
        results["length"][p] = defaultdict(list)
    #results = pickle.load(open("cache/random_hops/number_of_jumps/CS_SIS.p", "rb"))
    for trial in xrange(sir_trials):
        print("Trial progress: {}".format(trial / float(sir_trials)))
        for p in fjumps:
            print(p)
            for node in school_metadata.keys():
                epi = SIR(faculty_graph.copy(), p=0.1, r=0.2, is_random_jump=True, n_random_jumps=p, random_jump_p=0.001)
                epi.infect_node(node)
                epi.simulate()
                results["size"][p][node].append(epi.size)
                results["length"][p][node].append(epi.length)
    pickle.dump(results, open("cache/random_hops/number_of_jumps/CS_SIS.p", 'wb'))
    results.clear()

    print("SIS done")


def main():
    # set number of for each trials here (0 or more)
    run_trials_graph_with_random_hops(si_trials=50, sir_trials=25, sis_trials=25)


if __name__ == "__main__":
    main()
