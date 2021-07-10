# coding: utf-8
import argparse

def original_data_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../metr/')
    parser.add_argument('--data_file', default="matrix.npz")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--features', default=[1,1], type=list, help="1-4,the number of features")
    parser.add_argument('--data_steps', default="3-3", type=str, help="3-6")
    parser.add_argument('--data_types', default=["Q", "P"], type=list, help="1-4")
    parser.add_argument('--log_name', default="data_log", type=str)
    return parser

# 所有数据处理data_model.py都会有的
def common_para(parser):
    parser.add_argument('--dataset_dir', default='../metr/', type=str)
    parser.add_argument('--data_types', default=["Q", "P"], type=list)
    parser.add_argument('--data_steps', default="3-3", type=str, help="3-6")
    parser.add_argument('--proportion', default=1.0, type=float)
    parser.add_argument('--Normalize', type=int, default=1)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--data', default=1, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--test', default=1, type=int, help="Do test")
    parser.add_argument('--Times', default=1, type=int, help="the number of experiment")
    parser.add_argument('--data_log', default='data_log', type=str)
    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--Null_Val', type=float, default=0.0, help='value for missing data')

    parser.add_argument('--Batch_Size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=5, help="five learning rate")
    parser.add_argument('--min_learning_rate', type=float, default=2.0e-06)
    parser.add_argument('--base_lr', type=float, default=0.001,help='initial learning rate')
    parser.add_argument('--round', type=int, default=0, help='subway need round, metr and pems do not')
    parser.add_argument('--train_shuffle', type=int, default=0, help='train shuffle')

    parser.add_argument('--continue_train', type=int, default=0, help='initial withou old model')

    return parser


def RMASTGRU(parser):


    parser.add_argument('--H', default=2, type=int) # 1,2,3,4,5

    parser.add_argument('--Fe', default=64, type=int)
    parser.add_argument('--activations_gcn', default=[None, None], type=list)
    parser.add_argument('--units_gcn', default="64-64", type=str)#2,4,6,8,16
    parser.add_argument('--Ks_gcn', default=[2, 2], type=list)

    parser.add_argument('--model', default='RMASTGRU', type=str)
    parser.add_argument('--graph_pkl', default='adj_mx_metr.pkl', type=str)
    parser.add_argument('--D', default=256, type=int)# 16 32...256
    parser.add_argument('--T', default=288, type=int)
    parser.add_argument('--num_units', default='64-64', type=str) #16 ... 256
    parser.add_argument('--SE_file', default='SE.npz', type=str)
    parser.add_argument('--dw_file', default='deepwalk.npz', type=str)

    parser.add_argument('--filter_type', default='dual_random_walk', type=str)
    parser.add_argument('--La', default='L.npz', type=str)
    parser.add_argument('--Lr', default='Lr.npz', type=str)

    parser.add_argument('--ave_time', default='ave_time.xlsx', type=str)
    parser.add_argument('--Cl_Decay_Steps', default=2000, type=int)
    parser.add_argument('--Use_Curriculum_Learning', default=True, type=bool)
    parser.add_argument('--start_time', default=0 , type=int)
    parser.add_argument('--time_slot', default=5, type=int)
    parser.add_argument('--features', default=[1, 3], type=list, help="1-4,the number of features")
    return parser