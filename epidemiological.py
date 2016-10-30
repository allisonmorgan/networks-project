import networkx as nx
from numpy import *
import random

def simulate_SI_infection(g, p, n_iter):
    epidemic_size = []
    length = []
    edges_visited = {}

    for i in range(n_iter):
        # Initially, at time t = 0, all vertices are in the uninfected state
        susceptible = {}
        for node in g.nodes():
            susceptible[node] = True

        # To begin the epidemic, at time t = 1, choose a node uniformly at random
        # to infect.
        n_previously_infected = 0
        infected_node = random.randint(0, len(g.nodes())-1)
        infected = {}
        infected[infected_node] = True
        del susceptible[infected_node]
        t = 1

        # A simulated epidemic is deemed complete when no new infected nodes are
        # produced in a time step.
        edges_visited = {}
        while len(infected) > n_previously_infected:
            # Update the number of infected at this time step
            t += 1
            n_previously_infected = len(infected)
            new_infected = []

            # Loop through the infected nodes, looking for edges across which the 
            # infection could spread
            for node in infected:
                # Consider all neighbors of this node
                neighbors = g.neighbors(node)
                for neighbor in neighbors:
                    # If a neighbor is not infected, 
                    if infected.has_key(neighbor):
                        continue

                    # then flip a coin to determine if this node should get infected
                    edge = "{0} - {1}".format(neighbor, node)
                    opp_edge = "{0} - {1}".format(node, neighbor)
                    if not edges_visited.has_key(edge) and not edges_visited.has_key(opp_edge):
                        # Flip a coin. If heads, then infect this node
                        if random.random() <= p:
                            new_infected.append(neighbor)
                            if susceptible.has_key(neighbor):
                                del susceptible[neighbor]

                        edges_visited[edge] = True
                        edges_visited[opp_edge] = True

            # Add these newly infected nodes
            for each in new_infected:
                infected[each] = True

        epidemic_size.append(len(infected)/float(len(infected) + len(susceptible))) 
        length.append(t-1)

    return epidemic_size, length