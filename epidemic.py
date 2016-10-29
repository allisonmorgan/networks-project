import networkx as nx
import random


def flip(p):
    return True if random.random() <= p else False


class SI:
    """An SI epidemic model.

    Parameters
    ----------
    graph : the network the epidemic will run on

    p : transmission probability

    """
    def __init__(self, graph, p=0.5):
        self.p = p
        self.graph = graph
        self.susceptible = set(nx.nodes(graph))
        self.infected = set()
        self.time = 0
        self.is_complete = False
        self.visited = set()

    def __step(self):
        is_complete = True
        for u in self.infected.copy():
            nbrs = [v for v in self.graph.neighbors(u)
                    if v in self.susceptible and (u, v) not in self.visited]
            for v in nbrs:
                if flip(self.p):
                    self.susceptible.remove(v)
                    self.infected.add(v)
                    is_complete = False
                else:
                    self.visited.add((u, v))
        self.is_complete = is_complete
        if not self.is_complete:
            self.time += 1

    def simulate(self):
        self.time += 1
        random_node = random.choice(self.graph.nodes())
        self.susceptible.remove(random_node)
        self.infected.add(random_node)
        while not self.is_complete:
            self.__step()

    def size(self):
        return len(self.infected)


class SIR:
    """An SIR epidemic model.

    Parameters
    ----------
    graph : the network the epidemic will run on

    p : transmission probability
    r : recovery probability

    """
    def __init__(self, graph, p=0.5):
        self.p = p
        self.graph = graph
        self.susceptible = set(nx.nodes(graph))
        self.infected = set()
        self.recovered = set()
        self.time = 0
        self.is_complete = False
        self.visited = set()

    def __step(self):
        is_complete = True
        # Consider allowing infected nodes to recover
        for u in self.infected.copy():
            if flip(self.r):
                self.infected.remove(u)
                self.recovered.add(v)
        # Consider spreading the infection to the neighbors of all infected nodes
        for u in self.infected.copy():
            nbrs = [v for v in self.graph.neighbors(u)
                    if v in self.susceptible and (u, v) not in self.visited]
            for v in nbrs:
                if flip(self.p):
                    self.susceptible.remove(v)
                    self.infected.add(v)
                    is_complete = False
                else:
                    self.visited.add((u, v))
        self.is_complete = is_complete
        if not self.is_complete:
            self.time += 1

    def simulate(self):
        self.time += 1
        random_node = random.choice(self.graph.nodes())
        self.susceptible.remove(random_node)
        self.infected.add(random_node)
        while not self.is_complete:
            self.__step()

    def size(self):
        return len(self.infected)

    def length(self):
        return len(self.time)

