import networkx as nx
import random


# TODO: should move these utility functions somewhere else
def flip(p, weight=None):
    if weight == None:
        return True if random.random() <= p else False
    else:
        return True if random.random() <= p*float(weight) else False


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

    """
    def __init__(self, graph, p=0.5, random_start=True):
        self.p = p
        self.graph = graph
        self.susceptible = set(nx.nodes(graph))
        self.infected = set()
        self.time = 0
        self.visited_edges = set()
        self.is_complete = False

    def get_edge_weight(self, attributes):
        weight = None
        if attributes.has_key('weight'):
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
            edges_to_try = [(u, v, data) for (u, v, data) in 
                            self.graph.edges(u, data=True)
                            if v in self.susceptible
                            and (u, v) not in self.visited_edges]
            for u, v, data in edges_to_try:
                weight = self.get_edge_weight(data)
                self.visited_edges.add((u, v))
                if flip(self.p, weight):
                    self.infect_node(v)

    def step(self):
        if not self.is_complete:
            self.__infect_step()

    def simulate(self):
        while not self.is_complete:
            self.step()
            self.time += 1
        #print("time: {0}, length: {1}".format(self.size, self.length))

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
