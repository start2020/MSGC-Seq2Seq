# coding: utf-8
import numpy as np
import pandas as pd
import pickle

def excel_to_pkl():
    

    df = pd.read_excel("connection.xlsx")
    print(df)
    connection_s = df.values
    connection = connection_s[:, 1:]
    connection = connection.astype(np.float32)
    print(connection)
    f = open('adj_mx_subway.pkl', 'wb+')
    sensor_ids, sensor_id_to_ind = 1, 2
    pickle.dump([sensor_ids, sensor_id_to_ind, connection], f)
    f.close()
excel_to_pkl()