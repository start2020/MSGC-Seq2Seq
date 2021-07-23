import os
import numpy as np
import pandas as pd
from libs import utils, graph_common
import datetime
import sys
import tqdm


###################################### original_data.py ###################################
def load_original_data(args, log):
    data_file = os.path.join(args.dataset_dir, 'originals', args.data_file)
    M = np.load(data_file)['arr_0']
    utils.log_string(log, 'original data shape: ' + str(M.shape) + ' dtype: ' + str(
        M.dtype))
    return M




# 这个函数仅可以用于pems这种24小时都在运行的数据集
# (D,T1,N,F)=>(M,P,N,F),(M,Q,N,F)
def sequence(args, data):
    P, Q = int(str(args.data_steps.split("-")[0])), int(str(args.data_steps.split("-")[1]))
    D, T1, N, F = data.shape
    data1 = np.reshape(data, newshape=(D*T1, N, F))
    num_step, dims = data1.shape[0],N
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims, F),dtype=np.float32)
    y = np.zeros(shape = (num_sample, Q, dims, F),dtype=np.float32)
    for i in range(num_sample):
        x[i] = data1[i : i + P]
        y[i] = data1[i + P : i + P + Q]
    return [y,x]

def choose(type, data):
    if type == 1:
        output = data[...,0:1].astype(np.float32)
    if type == 2:
        output = data[...,0:2].astype(np.float32)
    if type == 3:
        output = data[...,0:3].astype(np.float32)
    if type == 4:
        output = data[...,0:4].astype(np.float32)
    return output

# (M,P,N,F),(M,Q,N,F) => (M,P,N,1),(M,Q,N,1)
def choose_features(args, log, data):
    labels = choose(int(args.features[0]), data[0])
    samples = choose(int(args.features[1]), data[1])
    utils.log_string(log, "samples shape:%s,labels shape:%s dtype:%s "% (samples.shape, labels.shape, samples.dtype))
    return [labels,samples]

def save_labels_samples(args, data, features):
    data_path = os.path.join(args.dataset_dir, args.model, utils.input_pattern(args))  # 路径
    labels_name = "labels_%s_%s.npz"%(str(args.data_steps.split("-")[0]), str(features[0]))
    samples_name = "samples_%s_%s.npz"%(str(args.data_steps.split("-")[0]), str(features[1]))
    data_names = [labels_name, samples_name]
    for i in range(len(data_names)):
        save_name = os.path.join(data_path, data_names[i])
        np.savez_compressed(save_name, data[i])

def save_TE(args, TE,features):
    data_path = os.path.join(args.dataset_dir, args.model, utils.input_pattern(args))  # 路径
    TE_name = "TE_%s_%s.npz"%(str(args.data_steps.split("-")[0]), str(features[1]))
    save_name = os.path.join(data_path, TE_name)
    np.savez_compressed(save_name, TE)

def standarize(samples, normalize=True):
    mean = np.mean(samples)
    std =  np.std(samples)
    if normalize:
        samples = (samples - mean) / std
    return samples, mean, std
###################################### data process ###############################


###################################### main.py ###################################


def data_process(args):
    log_dir, save_dir, data_dir = utils.create_dir_pattern(args)
    data_log, train_log, result_log = utils.create_logs(args, log_dir)
    utils.log_string(train_log, str(args)[10: -1])  # 打印超参数
    # 装载数据:Y=(M,Q,N,1),X=(M,P,N,1)
    data = load_labels_samples(args, data_log)
    # 标准化数据:只标准化输入X

    data[1], mean, std = standarize(data[1], normalize=args.Normalize)

    # 标准差方差
    file_name = os.path.join(data_dir, "mean_std.npz")
    np.savez_compressed(file_name, mean = mean, std = std)
    utils.log_string(data_log, "mean:{},std:{}".format(mean, std))

    if args.model in ['MSGC_Seq2Seq']:
        # adjacent matrix
        data_path = os.path.join(args.dataset_dir, "all_models", utils.input_pattern(args))  # 路径
        TE_name = "TE_%s_%s.npz" % (str(args.data_steps.split("-")[0]), str(args.features[1]))
        save_name = os.path.join(data_path, TE_name)
        TE = np.load(save_name)['arr_0'] #B,P+Q,2

        N = data[1].shape[2]
        P = data[1].shape[1]
        TE = np.tile(np.expand_dims(TE[:,0:P,:],axis=2), (1, 1, N, 1)) # (Days,T1,N,1)
        data[1] =np.concatenate([data[1],TE],axis=-1).astype(np.float32)

        graph_pkl_filename = os.path.join(args.dataset_dir, 'originals', args.graph_pkl)
        _, _, adj_mx = graph_common.load_graph_data(graph_pkl_filename)
        N,_ = adj_mx.shape
        adj_mx = adj_mx + np.identity(N, dtype=np.float32)
        La = adj_mx
        np.savez_compressed(os.path.join(data_dir, args.La), La)
        # prior matrix
        t = args.time_slot
        P = int(args.data_steps.split("-")[1])
        Q = int(args.data_steps.split("-")[0])
        df = pd.read_excel(os.path.join(args.dataset_dir, 'originals', args.ave_time), index_col=0)
        A = df.values
        A = A / 70000.0 * 60.0
        st_matrix = graph_common.prior(Q, P, A, t)

        #Lr = graph_common.all_transform(st_matrix, filter_type=args.filter_type)  # ndarray,(Q,P,N,N)
        Lr = st_matrix
        np.savez_compressed(os.path.join(data_dir, args.Lr), Lr)

        mat_filename = os.path.join(args.dataset_dir, 'originals', "matrix.npz")

        mat = np.load(mat_filename)['arr_0']  # (D,T,N,3)
        mat = mat[...,0]# (D,T,N)
        mat = mat[0:5]
        D,T,N = mat.shape
        mean = np.sum(np.sum(mat,axis=0),axis=0)/D/T #(N)
        weight = np.zeros(shape=(N,N),dtype=np.float32)
        for i in tqdm.trange(D):
            for j in range(T):
                for k in range(N):
                    for h in range(N):
                        if k == h: continue
                        if ((mat[i, j, k] > mean[k] and mat[i, j, h] > mean[h]) or
                            (mat[i, j, k] < mean[k] and mat[i, j, h] < mean[h])):
                            weight[k, h] += 1.0
                            weight[h, k] += 1.0
        weight = weight/D/T

        weight = weight + np.identity(N, dtype=np.float32)
        La = weight * La
        La = La-np.identity(N, dtype=np.float32)
        np.savez_compressed(os.path.join(data_dir, args.La), La)

    # 检查多维数组是否有nan,inf
    for d in data:
        utils.check_inf_nan(data_log, d)
    # 划分训练集/验证集/测试集
    data = data_split(args, data_log, data)
    # 打乱训练集
    data[0] = shuffle(data[0])
    # 划分batch
    for i in range(len(data)):
        data[i] = batch_split(args, data[i])
    # 保存数据
    save_dataslipt(args, data_dir, data)
    utils.log_string(data_log, 'Finish\n')

def batch_split(args, data):
    x = len(data)
    sample_num = data[0].shape[0]
    results=[]
    for i in range(x):
        batch_num = sample_num // args.Batch_Size
        sample_num = sample_num - sample_num % args.Batch_Size
        t = data[i][:sample_num, ...]
        t = np.stack(np.split(t, batch_num, axis=0), axis=0)
        results.append(t)
    return results

def save_dataslipt(args, data_dir, data):
    types = ['train', 'val', 'test']
    for i in range(len(types)):
        file_name = os.path.join(data_dir, types[i])
        if args.model in ["MSGC_Seq2Seq"]:
            np.savez_compressed(file_name, data[i][0], data[i][1])
        else:
            raise ValueError

def data_split(args, log, data):
    batch_num = data[0].shape[0]
    train_num = round(args.train_ratio * batch_num)
    val_num = round(args.val_ratio * batch_num)

    x = len(data)
    trains = []
    vals = []
    tests = []


    for i in range(x):
        train = data[i][:train_num, ...]
        val = data[i][train_num:train_num+val_num, ...]
        test = data[i][train_num+val_num:, ...]


        trains.append(train)
        vals.append(val)
        tests.append(test)
        utils.log_string(log, 'data[%s]=>train: %s\tval: %s\ttest: %s' % (i, train.shape, val.shape, test.shape))
    return [trains, vals, tests]


def proportion(prop, log, data):
    num_sample = int(np.floor(data[0].shape[0] * prop))
    for i in range(len(data)):
        data[i] = data[i][:num_sample,...]
        utils.log_string(log, 'data[%s]=>shape: %s' % (i, data[i].shape))
    return data

def shuffle(data):
    results = []
    sample_num = data[0].shape[0]
    #per = list(np.random.RandomState(seed=42).permutation(sample_num)) # 固定seed
    per = list(np.random.permutation(sample_num))
    for i in range(len(data)):
        results.append(data[i][per,...])
    return results

def load_labels_samples(args, log):
    path = os.path.join(args.dataset_dir, 'all_models',  utils.input_pattern(args))
    results = []
    labels_name = "labels_%s_%s.npz"%(str(args.data_steps.split("-")[0]), str(args.features[0]))
    samples_name = "samples_%s_%s.npz"%(str(args.data_steps.split("-")[1]), str(args.features[0]))
    data_names = [labels_name, samples_name]
    for data_name in data_names:
        file_path = os.path.join(path, data_name)
        result = np.load(file_path)['arr_0']
        utils.log_string(log, '%s %s' % (data_name, str(result.shape)))
        results.append(result)
    return results


############################################ main ################################################
def load_data(args, log, data_dir):
    utils.log_string(log, 'loading data...')
    types = ['train','val','test']
    results = []
    for type in types:
        path = os.path.join(data_dir, '%s.npz'%(type))
        data = np.load(path)
        dict = {}
        for j in range(len(data)):
            name = "arr_%s" % j
            dict[name] = data[name]
            utils.log_string(log, '%s=>%s shape: %s,type:%s' % (type, j, dict[name].shape,dict[name].dtype))
        results.append(dict)
    # 装载标准差和均值
    if args.Normalize:
        path = os.path.join(data_dir, 'mean_std.npz')
        data = np.load(path)
        mean, std = data['mean'], data['std']
    else:
        mean, std = 0.0, 1.0 # 数据不标准化
    utils.log_string(log,'mean:{},std:{}'.format(mean, std))
    return results, mean, std


