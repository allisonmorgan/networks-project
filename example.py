import matplotlib.pyplot as plt
import numpy as np

from epidemic import SI, SIR, SIS
from network import bi_planted_partition


def main():
    # example
    n_networks = 500
    x = np.linspace(0, 1, 50, endpoint=False)
    results_b_length = []
    results_b_size = []
    for p in x:
        print(p)
        samples_length = []
        samples_size = []
        for i in range(n_networks):
            #print('-- NEW EPI --')
            g = bi_planted_partition(1000, 8, 0)
            epi = SIS(g, p=p, r=0.04)
            #epi = SI(g, p=p)
            epi.infect_random_node()
            epi.simulate()
            samples_length.append(epi.length)
            samples_size.append(epi.size)
        results_b_length.append((p, np.average(samples_length)))
        results_b_size.append((p, np.average(samples_size)))

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(*zip(*results_b_length))
    plt.xlabel('Transmission probability')
    plt.ylabel('Epidemic length')
    plt.axhline(y=np.log(1000), linewidth=1, color='red')
    plt.show()
    #plt.savefig('length.png')
    plt.clf()

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(*zip(*results_b_size))
    plt.xlabel('Transmission probability')
    plt.ylabel('Epidemic size')
    plt.show()
    #plt.savefig('size.png')

if __name__ == "__main__":
    main()
