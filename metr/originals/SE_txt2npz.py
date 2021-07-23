# coding: utf-8
import numpy as np
import pandas as pd

def changemode():
    file_name = "SE(METR).txt"
    fin = open(file_name)
    line = fin.readline()[:-1]
    print(line)
    N = int(line.split(" ")[0])
    dimensions = int(line.split(" ")[1])
    print()
    SE_mat = np.zeros(shape=[N,dimensions])
    # se_mat = []
    for line in fin.readlines():
        line = line[:-1]
        line_split = line.split(" ")
        # print(line_split)
        # print(len(line_split))
        # exit()
        tmp = []
        for i in range(dimensions):
            tmp.append(float(line_split[i+1]))
        index = int(line_split[0])
        # se_mat.append(tmp)
        SE_mat[index] = tmp

    np.savez_compressed("SE.npz",SE=np.array(SE_mat))

changemode()

