import pandas as pd
import numpy as np
import datetime


traffic_file = "METR.h5"
df = pd.read_hdf(traffic_file, key='df')
data = df.values
print(type(data))
print(data.shape)
T1 = int(24*60/5)
days = data.shape[0]//T1
print(T1,days)

Time = df.index
print(Time[-3:])
N = data.shape[-1]
dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
timeofday = (Time.hour * 60 + Time.minute + Time.second / 60) // 5
timeofday = np.reshape(timeofday, newshape = (-1, 1))
dayofyear = np.reshape(Time.dayofyear-61, newshape = (-1, 1))
new_time = np.concatenate((timeofday,dayofweek,dayofyear), axis = -1) #(days*T1, 2)

new_time = np.expand_dims(new_time, axis=1)  # (days*T1, 1, 2)
new_time = np.tile(new_time,(1,N, 1)) # (days*T1, N, 2)
print(new_time.shape)
data = np.expand_dims(data, axis=-1)  # (days*T1, N, 1)
print(data.shape)
data = np.concatenate([data,new_time ], axis=-1)#(Days*T1,N,3)
print(data.shape)
# data_file = './matrix-gman.npz'
# np.savez_compressed(data_file, data)
data = data.reshape(days,T1,N,4)
data = data.astype(np.float32)
print(data.dtype)
print(data.shape)
data_file = './matrix.npz'
np.savez_compressed(data_file, data)# (Days,T1,N,3)

#
# data = np.expand_dims(np.sum(np.sum(data, axis=-1), axis=-1), axis=-1)  # (days, T1, N, 1)
# print(data.shape)
# date_file = './day-index-201504.txt'
# dayofweek = week_transform(date_file)  # 时间特征，(Days,)
# data = add_time(data, dayofweek)
# print(data.shape)  # (Days,T1,N,3) (31, 64, 118, 3)
#
# print(data.shape)
# data_file = './matrix.npz'
# np.savez_compressed(data_file, data)


# days, t1, n,f = data.shape
# data = np.reshape(data, newshape=(days*t1, n, f ))
# print(data.shape)
# data_file = './matrix-gman.npz'
# np.savez_compressed(data_file, data)