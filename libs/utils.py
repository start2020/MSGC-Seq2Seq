import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import datetime

def print_parameters(log):
    parameters = 0
    for variable in tf.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

'''
write information to the log and print it
'''
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# 检查多维数组中是否存在nan,inf
def check_inf_nan(log, Ms):
    nan_num = np.sum(np.isnan(Ms).astype(np.float32))
    inf_num = np.sum(np.isinf(Ms).astype(np.float32))
    log_string(log, "Number of nan:{},Number of inf:{}".format(nan_num, inf_num))

'''
create a string for each input pattern
'''
def input_pattern(args):
    pattern = ''
    print(args.data_steps)
    data_steps = args.data_steps.split("-")
    print(data_steps,len(args.data_types))
    for i in range(len(args.data_types)):
        pattern += str(args.data_types[i]) + data_steps[i]
    return pattern


def create_path(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


'''
create the dir to save processed data under each input pattern
create the log
'''
def create_dir_original(args):
    data_path = os.path.join(args.dataset_dir, args.model, input_pattern(args))
    create_path(data_path)
    return data_path


'''
create dir for all models under each input pattern
'''
def create_dir_pattern(args):
    dir = os.path.join(args.dataset_dir, input_pattern(args))
    log_dir = os.path.join(dir, 'log', args.model)
    save_dir = os.path.join(dir, 'save', args.model)
    data_dir = os.path.join(dir, 'data', args.model)
    create_path(save_dir)
    create_path(log_dir)
    create_path(data_dir)
    return log_dir, save_dir, data_dir


def create_log(log_dir, log_name):
    log_file = os.path.join(log_dir, log_name)
    log = open(log_file, 'a')
    return log


def create_logs(args, log_dir):
    data_log = create_log(log_dir, args.data_log)
    train_log = create_log(log_dir, args.train_log)
    result_log = create_log(log_dir, args.result_log)
    return data_log, train_log, result_log