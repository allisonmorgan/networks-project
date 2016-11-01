import networkx as nx


def bi_planted_partition(n, c, eps):
    p_in = (2.*c + eps) / (2*n)
    p_out = (2.*c - eps) / (2*n)
    return nx.random_partition_graph([n/2, n/2], p_in, p_out)
