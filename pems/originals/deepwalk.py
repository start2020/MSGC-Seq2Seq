
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import networkx as nx
from sklearn.manifold import TSNE



if __name__ == "__main__":

    graph_file_name = 'my_graph.txt'
    graph_f = open(graph_file_name, 'w', encoding='utf8')

    seq_len = 7 * 24 * 60 // 5
    for i in range(seq_len-1):
        graph_f.write("%d %d\n"%(i,(i+1)))
    graph_f.close()
    G = nx.read_edgelist(graph_file_name,
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)]) #无线图,节点权重为1

    model = DeepWalk(G, walk_length=12, num_walks=180, workers=1)
    model.train(window_size=10, iter=100,embed_size=64)
    # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    # model.train(window_size=5, iter=3,embed_size=128)
    embeddings = model.get_embeddings()
    print()
    print(model.w2v_model.total_train_time)
    embedding_npz = []
    for i in range(seq_len):
        embedding_npz.append(embeddings[str(i)])
    np.savez_compressed("deepwalk.npz",SE=embedding_npz)
    print("finish")


