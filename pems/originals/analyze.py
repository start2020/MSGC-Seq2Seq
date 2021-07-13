import pandas as pd
import numpy as np
import datetime


# traffic_file = "PeMS.h5"
# df = pd.read_hdf(traffic_file)
#
# data = df.values
# # tmp_df = df[0]
# # tmf_file_name = "tmp.xlsx" #保存后删除第一列
# # tmp_df.to_excel()
# # print(tmp_df)
#
# new_pf = pd.read_excel('./tmp.xlsx', sheet_name = 0)
# Time = df.index
#
# print(new_pf)
#
# T1 = int(24*60/5)
#
# start_time = "2017-01-01 00:00:00"
# start_time_dt = datetime.datetime.strptime(start_time,"%Y-%m-%d %H:%M:%S")
# new_pf.loc[start_time_dt] = data[0]
# loss_index_num = 0
# for time_index in range(1,df.shape[0]):
#     print(time_index/df.shape[0])
#     time_delta = Time[time_index] - Time[time_index - 1]
#     seconds = time_delta.seconds
#     if seconds == 300:  # 5分钟
#         cur_time = str((start_time_dt + datetime.timedelta(minutes=(time_index+loss_index_num) * 5)).strftime("%Y-%m-%d %H:%M:%S"))
#         new_pf.loc[datetime.datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")] = data[time_index]
#     else:
#         err_index = 0
#         print(seconds)
#         k = seconds//300 #一次补全k个数据
#
#         for j in range(k):
#             cur_time = str((start_time_dt + datetime.timedelta(minutes=(time_index + loss_index_num + j) * 5)).strftime(
#                 "%Y-%m-%d %H:%M:%S"))
#             res = new_pf.values[(time_index + loss_index_num+ j)-T1*7]#用上一周数据来填补丢失的数据
#             new_pf.loc[datetime.datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")] = res
#         loss_index_num += k
#
# print(new_pf.shape)
#
#
# output_name = "pems_c.h5"
# new_pf.to_hdf(output_name,'obj3',format='table')
# df = pd.read_hdf(output_name)
# print(df.values.shape)

traffic_file = "pems_c.h5"
new_pf = pd.read_hdf(traffic_file)


T1 = int(24*60/5)
print("T1 ",T1)
Time = new_pf.index
data = new_pf.values

N = data.shape[-1]
days = data.shape[0]//T1
dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
timeofday = (Time.hour * 60 + Time.minute + Time.second / 60) // 5
timeofday = np.reshape(timeofday, newshape = (-1, 1))
dayofyear = np.reshape(Time.dayofyear-1, newshape = (-1, 1))
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

