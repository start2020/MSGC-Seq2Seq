import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
# 对整个图采样100次
num_walks = 100
# 每个walk的长度是80个节点
walk_length = 80
# 每个点的维度是64
dimensions = 64
# 将每个点的上下文点限制在10个以内
window_size = 10

iter = 1000
Adj_file = 'Adj.txt'
SE_file = 'SE.txt'

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size = dimensions, window = 10, min_count=0, sg=1,
        workers = 8, iter = iter)
    model.wv.save_word2vec_format(output_file)
    
nx_G = read_graph(Adj_file)

G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
