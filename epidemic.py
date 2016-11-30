import random
from collections import defaultdict

import networkx as nx


# TODO: should move these utility functions somewhere else
def flip(p, weight=None):
    if not weight:
        return True if random.random() <= p else False
    else:
        return True if float(random.random()) <= p*float(weight) else False


def filter_from(edges, from_node):
    """Remove edges (from `edges`) that start at `from_node`

    """
    return set([edge for edge in edges if edge[0] != from_node])


class SI(object):
    """An SI epidemic model.

    Parameters
    ----------
    graph : the network the epidemic will run on

    p : transmission probability

    is_random_jump : for each node u, if that node ever gets infected,
        give it exactly n_random_jumps (see below) chance to jump to a random node
        in a random SCC distinct from the one that node u is a member of

    random_jump_p : probability of a random jump (if is_random_jump is enabled)
        happening

    n_random_jumps : number of random jumps to try if is_random_jump enabled

    """
    def __init__(self,
                 graph,
                 p=0.5,
                 is_random_jump=False,
                 random_jump_p=0.001,
                 n_random_jumps=1):
        self.p = p
        self.graph = graph
        self.susceptible = set(nx.nodes(graph))
        self.infected = set()
        self.time = 0
        self.visited_edges = set()
        self.is_complete = False
        self.is_random_jump = is_random_jump
        self.attempted_random_jump = defaultdict(bool)
        self.random_jump_p = random_jump_p
        self.n_random_jumps = n_random_jumps
        # TODO: descendents should be memoized across different epidemics for the same network
        self.descendents = {}
        if is_random_jump:
            for u in self.graph.nodes():
                self.descendents[u] = nx.descendants(self.graph, u)

    def get_edge_weight(self, edge):
        weight = None
        attributes = self.graph.get_edge_data(edge[0], edge[1])
        if 'weight' in attributes:
            weight = attributes['weight']
        return weight

    def infect_random_node(self):
        try:
            random_node = random.choice(list(self.susceptible))
            self.infect_node(random_node)
        except:
            print("No susceptible nodes to infect.")

    def infect_node(self, node):
        """Infect a node if it is susceptible.

        """
        try:
            self.susceptible.remove(node)
            self.infected.add(node)
            self.is_complete = False
        except:
            pass
            #print("Node {} is not susceptible.".format(node))

    def __infect_step(self):
        # The epidemic is complete if time passed,
        # but the infection didn't spread.
        self.is_complete = True
        for u in self.infected.copy():
            edges_to_try = [e for e in self.graph.edges(u)
                            if e[1] in self.susceptible
                            and e not in self.visited_edges]
            for e in edges_to_try:
                weight = self.get_edge_weight(e)
                self.visited_edges.add(e)
                if flip(self.p, weight):
                    self.infect_node(e[1])
            if self.is_random_jump and not self.attempted_random_jump[u]:
                self.attempted_random_jump[u] = True
                nodes_of_graph = set(self.graph.nodes())
                reachable_from_u = self.descendents[u]
                unreachable_from_u = nodes_of_graph.difference(reachable_from_u)
                susc_unreachable_from_u = unreachable_from_u.difference(self.infected)
                if not susc_unreachable_from_u:
                    continue
                vs = random.sample(susc_unreachable_from_u, self.n_random_jumps)
                for v in vs:
                    if flip(self.random_jump_p):
                        self.infect_node(v)

    def step(self):
        if not self.is_complete:
            self.__infect_step()

    def simulate(self):
        while not self.is_complete:
            self.step()
            self.time += 1

    @property
    def size(self):
        return len(self.infected)/float(self.graph.number_of_nodes())

    @property
    def length(self):
        return self.time


class SIR(SI):
    def __init__(self, graph, p=0.2, r=0.8):
        super(SIR, self).__init__(graph, p)
        self.r = r
        self.recovered = set()

    def __recover_step(self):
        for u in self.infected.copy():
            if flip(self.r):
                self.infected.remove(u)
                self.recovered.add(u)

    def step(self):
        self.__recover_step()
        super(SIR, self).step()


class SIS(SI):
    def __init__(self, graph, p=0.2, r=0.8):
        super(SIS, self).__init__(graph, p)
        self.r = r

    def __recover_step(self):
        for u in self.infected.copy():
            if flip(self.r):
                self.infected.remove(u)
                self.susceptible.add(u)
                # TODO: Should the following be uncommented?
                # self.visited_edges = filter_from(self.visited_edges, u)

    def step(self):
        self.__recover_step()
        super(SIS, self).step()
