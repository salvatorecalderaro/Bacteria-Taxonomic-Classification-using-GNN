import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import torch

alphabet="ACGT"

def to_categorical(y):
    return np.eye(len(alphabet), dtype='uint8')[y]

def one_hot_encoding(seq):
    mp = dict(zip(alphabet, range(len(alphabet))))
    seq_2_number = [mp[nucleotide] for nucleotide in seq]
    return to_categorical(seq_2_number).flatten()


def get_kmer_count_from_sequence(sequence, k, cyclic=True):
    """
    Returns dictionary with keys representing all possible kmers in a sequence
    and values counting their occurrence in the sequence.
    """
    # dict to store kmers
    kmers = {}

    # count how many times each occurred in this sequence (treated as cyclic)
    for i in range(0, len(sequence)):
        kmer = sequence[i:i + k]

        # for cyclic sequence get kmers that wrap from end to beginning
        length = len(kmer)
        if cyclic:
            if len(kmer) != k:
                kmer += sequence[:(k - length)]

        # if not cyclic then skip kmers at end of sequence
        else:
            if len(kmer) != k:
                continue

        # count occurrence of this kmer in sequence
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1

    return kmers

def get_debruijn_edges_from_kmers(kmers):
    """
    Every possible (k-1)mer (n-1 suffix and prefix of kmers) is assigned
    to a node, and we connect one node to another if the (k-1)mer overlaps 
    another. Nodes are (k-1)mers, edges are kmers.
    """
    # store edges as tuples in a set
    edges = set()

    # compare each (k-1)mer
    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2:
                # if they overlap then add to edges
                if k1[1:] == k2[:-1]:
                    edges.add((k1[:-1], k2[:-1]))
                if k1[:-1] == k2[1:]:
                    edges.add((k2[:-1], k1[:-1]))

    return edges

def create_graph(sequence, k):
    features = []
    kmers = get_kmer_count_from_sequence(sequence, k, cyclic=False)
    e = get_debruijn_edges_from_kmers(kmers)

    g = nx.from_edgelist(e)
    for n in g.nodes:
        c = one_hot_encoding(n)
        features.append(c)

    features = np.stack(features).astype(np.float32)
    G=from_networkx(g)
    G["x"]=torch.tensor(features)
    
    return G