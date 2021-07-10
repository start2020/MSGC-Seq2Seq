import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from libs import utils, data_common,para
import numpy as np

def main(args):
    data_path = utils.create_dir_original(args)
    log = utils.create_log(data_path, args.log_name)
    utils.log_string(log, str(args)[10: -1])  # 打印超参数
    # 原始数据格式是NPZ,(D,T1,N,F)
    M = data_common.load_original_data(args, log)
    # # (D,T1,N,F)=>(M,P,N,F),(M,Q,N,F)
    M = data_common.sequence(args, M)


    #gman,取day of week 和 day of time
    features= [1,3]
    TE_Q = M[0][...,features[0]:features[1]] #(M,Q,N,2)
    TE_P = M[1][...,features[0]:features[1]] #(M,P,N,2)
    TE = np.concatenate([TE_P,TE_Q], axis=1)[...,0,:]
    print(TE.shape, TE.dtype)
    time_of_day = TE[...,0:1]
    day_of_week = TE[..., 1:2]
    Time = np.concatenate([day_of_week,time_of_day],axis=-1)
    Time = Time.astype(np.int32)
    data_common.save_TE(args, Time,features)

    features = [1, 4]
    TE_Q = M[0][..., features[0]:features[1]]  # (M,Q,N,2)
    TE_P = M[1][..., features[0]:features[1]]  # (M,P,N,2)
    TE = np.concatenate([TE_P, TE_Q], axis=1)[..., 0, :]
    print(TE.shape, TE.dtype)
    time_of_day = TE[..., 0:1]
    day_of_week = TE[..., 1:2]
    day_of_year = TE[..., 2:3]
    Time = np.concatenate([day_of_week, time_of_day,day_of_year], axis=-1)
    Time = Time.astype(np.int32)
    data_common.save_TE(args, Time, features)


    # DCRNN 取day of time
    features = [1, 2]
    TE_P = M[1][..., 1:2]  # (M,P,N,2)
    TE = TE_P
    TE = TE.astype(np.int32)
    print(TE.shape, TE.dtype)
    data_common.save_TE(args, TE, features)

    features = [1, 1]
    M = data_common.choose_features(args, log, M)
    data_common.save_labels_samples(args, M, features)



if __name__ == "__main__":
        parser = para.original_data_para()
        args = parser.parse_args()
        main(args)