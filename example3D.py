import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from epidemic import SI, SIR, SIS
from network import bi_planted_partition

def main():
    n_networks = 100

    ps = np.linspace(0, 1, 25, endpoint=False)
    rs = np.linspace(0, 1, 25, endpoint=False)
    
    results_p = []; results_r = []; results_s = []; results_l = []
    for r in rs:
        for p in ps:
            print (r, p)
            samples_l = []
            samples_s = []
            for i in range(n_networks):
                #print('-- NEW EPI --')
                g = bi_planted_partition(1000, 8, 0)
                epi = SIS(g, p=p, r=r)
                #epi = SI(g, p=p)
                    
                epi.infect_random_node()
                epi.simulate()

                samples_l.append(epi.length)
                samples_s.append(epi.size)

            results_r.append(r)
            results_p.append(p)
            results_s.append(np.average(samples_s))
            results_l.append(np.average(samples_l))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = results_r, results_p
    Z = results_l
    ax.scatter(X, Y, Z)

    ax.set_xlabel('Recovery Probability')
    ax.set_ylabel('Transmission Probability')
    ax.set_zlabel('Epidemic Length')

    plt.show()
    # TODO: Figure out the right way to save these plots
    plt.clf()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = results_r, results_p
    Z = results_s
    ax.scatter(X, Y, Z)

    ax.set_xlabel('Recovery Probability')
    ax.set_ylabel('Transmission Probability')
    ax.set_zlabel('Epidemic Size')

    plt.show()
    # TODO: Figure out the right way to save these plots
    plt.clf()

if __name__ == "__main__":
    main()
