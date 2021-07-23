import node2vec
import networkx as nx
import numpy as np

Adj_file = 'Adj.txt'
output_file = 'SE.txt'

# Adj_file = 'ODAverageTime_hz_word2vec'
# output_file = 'SE-ave_time.txt'

walk_length = 80 #随意
dimensions = 64 # 每一个节点的特征长度
p = 2 #随意
nx_G = nx.read_edgelist(Adj_file, nodetype=int, data=(('weight',float),),create_using=nx.DiGraph())
node2vec_model = node2vec.Node2Vec(nx_G,dimensions=dimensions,walk_length=walk_length,p=p,workers=4)
model = node2vec_model.fit()
model.wv.save_word2vec_format(output_file)
np_res = model.wv.vectors #这个结果是乱序的,顺序在key_to_index里
key_res = model.wv.key_to_index
res = []
for i in range(np_res.shape[0]):
    res.append(np_res[key_res[str(i)]])
res = np.array(res,dtype=np.float32)
np.savez_compressed("SE.npz", SE=res)
print(np_res)

