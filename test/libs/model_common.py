# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


'''
逆标准化层
'''
def inverse_positive_1(X, std, mean, unit):
    inputs = X * std + mean
    inputs = fc_layer(inputs, unit)
    inputs = activation_layer(inputs, activation='relu')
    return inputs

def inverse_positive(X, std, mean):
    inputs = X * std + mean
    inputs = activation_layer(inputs, activation='relu')
    return inputs

def choose_RNNs(unit,type='GRU'):
    if type=="GRU":
        cell = tf.nn.rnn_cell.GRUCell(num_units=unit)
    elif type=="LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=unit, state_is_tuple=False)
    elif type =="RNN":
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=unit)
    else:
        print("Wrong type")
        cell = None
    return cell

'''
RNN 层
输入：(B,P,F)
输出: (B,F)
'''
def lstm_layer(X, unit,type='GRU'):
    cell  = choose_RNNs(unit=unit, type=type)
    outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
    output = outputs[:, -1, :] #(B,F)
    return output

def multi_lstm(X, units, type='GRU'):
    cells = [choose_RNNs(unit=unit, type=type) for unit in units]
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_cells, inputs=X, dtype=tf.float32)
    # output = outputs[:, -1, :] #(B,F)
    return outputs

def multi_cells(units, type='GRU'):
    cells = [choose_RNNs(unit=unit, type=type) for unit in units]
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    return stacked_cells

'''
多层FC
'''
def multi_fc(X, activations=['relu'], units=[64], drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        inputs = fc_layer(inputs, units=units[i])
        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

'''
多层GCN
'''
def multi_gcn(L, X, activations=["relu"], units=[128], Ks = [None],drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        inputs = gcn_layer(L, inputs, K=Ks[i])
        inputs = fc_layer(inputs,units=units[i])
        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

'''
空间静态嵌入
针对(B,T,N,N)，给予每个点一个spatial embedding
输入：(N,Fs)
输出: (1,1,N,D),D是对齐维度
'''
def s_embbing_static(SE, D, activations=['relu',None], drop_rate=None, bn=False, dims=None, is_training=True):
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0) # (1,1,N,Fs)
    SE = multi_fc(SE, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    return SE # (1,1,N,D)

'''
时间静态嵌入
针对(B,T,N,N)，给予每个点一个spatial embedding
输入：(N,Fs)
输出: (B,T,N,D),D是对齐维度
'''
# def t_embbing_static(t, TE, T, D, activations=['relu',None], drop_rate=None, bn=False, dims=None, is_training=True):
#     if t==0:
#         TE = tf.one_hot(TE[..., 0], depth=7)  # (B,T,N,7),dayofweek
#     elif t==1:
#         TE = tf.one_hot(TE[..., 1], depth=T)  # (B,T,N,T)
#     elif t==2:
#         dayofweek = tf.one_hot(TE[..., 0], depth=7)
#         timeofday = tf.one_hot(TE[..., 1], depth=T)  # (B,T,N,T)
#         TE = tf.concat((dayofweek, timeofday), axis=-1)  # (B,T,N,T+7)
#     else:
#         print("No time feature")
#     TE = tf.cast(TE,tf.float32) #(B,P,N,2)
#     TE = multi_fc(TE, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
#     return TE
def t_embbing_static(c, t, TE, T, D, activations=['relu', None], drop_rate=None, bn=False, dims=None, is_training=True):
    # if t==0:
    #     TE = tf.one_hot(TE[..., 1], depth=7)  # (B,T,N,7),dayofweek
    # elif t==1:
    #     TE = tf.one_hot(TE[..., 0], depth=T)  # (B,T,N,T)
    # elif t==2:
    #     dayofweek = tf.one_hot(TE[..., 1], depth=7)
    #     timeofday = tf.one_hot(TE[..., 0], depth=T)  # (B,T,N,T)
    #     TE = tf.concat((dayofweek, timeofday), axis=-1)  # (B,T,N,T+7)
    # else:
    #     print("No time feature")
    # TE = tf.cast(TE,tf.float32) #(B,P,N,2)

    dayofweek = TE[..., 0]  # (B,T,N)
    timeofday = TE[..., 1]  # (B,T,N)
    indexes = dayofweek * 24 + timeofday  # (B,T,N)
    TE = tf.nn.embedding_lookup(c, indexes)  # (B,T,N,F)

    TE = multi_fc(TE, activations=activations, units=[D, D], drop_rate=drop_rate, bn=bn, dims=dims,
                  is_training=is_training)
    return TE



'''
特征拼接
X:(B,P,N,N)=>(B,P,N,D)
SE:(1,1,N,D)
TE:(B,P,N,D)
输出：(B,P,N,2D)
'''
def x_embedding(X, D, activations=['relu',None],drop_rate=None, bn=False, dims=None, is_training=True):
    X = multi_fc(X, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    return X

# def x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=True):
#     type = [is_X, is_SE, is_TE]
#     print(SE,TE,X)
#     if type == [True, True, True]:
#         STE = tf.add(SE, TE)  # (B,P,N,D)
#         STE = tf.concat((SE, TE),axis=-1)  # (B,P,N,D)
#         output = tf.concat((X, STE), axis=-1)  # (B,P,N,2D)
#     elif type == [False, True, True]:
#         #STE = tf.add(SE, TE)  # (B,P,N,D)
#         STE = tf.concat((SE, TE),axis=-1)
#         output = STE
#     elif type == [True, False, False]:
#         output = X
#     else:
#         print("Wrong Type!")
#     return output

def x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=True):
    # print("X ",X)
    # print("SE ",SE)
    # print("TE ",TE)

    type = [is_X, is_SE, is_TE]
    if type == [True, True, True]:
        STE = tf.add(SE, TE)  # (B,P,N,D)
        output = tf.concat((X, STE), axis=-1)  # (B,P,N,2D)
    elif type == [False, True, True]:
        STE = tf.add(SE, TE)  # (B,P,N,D)
        output = STE
    elif type == [True, True, False]:
        # print("no TE")
        tmp = tf.zeros_like(X)
        SE = tf.add(SE, tmp)  # (B,P,N,D)
        output = tf.concat((X, SE), axis=-1)  # (B,P,N,2D)

    elif type == [True, False, True]:
        # print("no SE")
        output = tf.concat((X, TE), axis=-1)  # (B,P,N,2D)

    elif type == [True, False, False]:
        output = X
    else:
        raise ValueError
        print("Wrong Type!")
    print("output",output)
    # exit()
    return output

def x_spatio_temporal(X, SE, TE, activations=['relu',None],drop_rate=None, bn=False, dims=None, is_training=True):
    D = SE.shape[-1]
    STE =tf.add(SE, TE) # (B,P,N,D)
    X = multi_fc(X, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    X = tf.concat((X, STE), axis=-1) #(B,P,N,2D)
    return X

'''
conv2d 层(无激活)
输入:[..., input_dim], 输出[..., output_dim]
'''
def conv2d_layer(x, output_dims, kernel_size, stride=[1, 1],padding='SAME'):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    # 卷积核用glorot_uniform初始化
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape=kernel_shape),
        dtype=tf.float32, trainable=True, name='kernel')
    # 卷积
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding=padding)
    bias = tf.Variable(tf.zeros_initializer()(shape=[output_dims]),dtype=tf.float32, trainable=True, name='bias')
    x = tf.nn.bias_add(x, bias)
    return x

'''
线性变换层
输入(...,F)，输出(...,units)
'''
def fc_layer(X, units=128):
    W = tf.Variable(tf.glorot_uniform_initializer()(shape = [X.shape[-1], units]),
                    dtype = tf.float32, trainable = True, name='kernel') #(F, F1)
    b = tf.Variable(tf.zeros_initializer()(shape = [units]),
                    dtype = tf.float32, trainable = True, name = 'bias') #(F1,)
    Y = tf.matmul(X, W) + b #(...,F)*(F,F1)+(F1,)=>(...,F1)
    return Y

def fc(X, output_unit):
    with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [X.shape[-1], output_unit], dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer())
        bias = tf.get_variable('bias', [output_unit], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        wxb = tf.matmul(X, weights)+ bias  # (B,N,F)*(F,S) => (B,N,S) +  (S) =>  (B,N,S)
        return wxb

''''
功能：可是输入静态矩阵，也可以输入动态矩阵，进行图卷积
K = None，表示无切比雪夫公式；K = 4 表示4阶切比雪夫公式
L=(N,N),X=(B,N,F),Output=(B,N,F1)
L=(B,T,N,N),X=(B,T,N,F),Output=(B,T,N,F1)
L=(N,N),X=(B,T,N,F),Output=(B,T,N,F1)
'''
def gcn_layer(L, X, K = None):
    y1 = X
    y2 = tf.matmul(L,y1)
    if K:
        x0, x1 = y1, y2
        total = [x0, x1]
        for k in range(3, K + 1):
            x2 = 2 * tf.matmul(L,x1) - x0
            total.append(x2)
            x1, x0 = x2, x1
        total = tf.concat(total, axis=-1)
        y2 = total
    return y2

'''
激活层
带有norm层
'''
def activation_layer(X, activation='relu', bn=False, dims=None, is_training=True):
    inputs = X
    if activation!=None:
        if bn:
            inputs = batch_norm(inputs, dims, is_training)
        if activation == 'relu':
            inputs = tf.nn.relu(inputs)
        elif activation == 'sigmoid':
            inputs = tf.nn.sigmoid(inputs)
        elif activation == 'tanh':
            inputs = tf.nn.tanh(inputs)
    return inputs

'''
Dropout 层
训练时，采用dropout，测试不采用
'''
def dropout_layer(x, drop_rate, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate=drop_rate),
        lambda: x)
    return x

'''
标准化层
'''
def batch_norm(x, dims, is_training):
    # 形状
    shape = x.get_shape().as_list()[-dims:]
    # 偏置系数，初始值为0
    beta = tf.Variable(
        tf.zeros_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='beta')
    # 缩放系数，初始值为1
    gamma = tf.Variable(
        tf.ones_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='gamma')
    # 计算均值和方差
    moment_dims = list(range(len(x.get_shape()) - dims))
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(0.9)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # 标准化:x输入，mean均值，var方差，beta=偏移值，gama=缩放系数，1e-3=防止除零
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

'''
所有可训练参数，加入正则化
'''
def regularizer(Loss):
    tf.add_to_collection(name="losses", value=Loss)
    for variable in tf.trainable_variables():
        tf.add_to_collection(name='losses', value=tf.nn.l2_loss(variable))
    Loss_l2 = tf.add_n(tf.get_collection("losses"))
    return Loss_l2

############################################### GMAN #############################################

def conv2d(x, output_dims, kernel_size, stride = [1, 1],
           padding = 'SAME', use_bias = True, activation = tf.nn.relu,
           bn = False, bn_decay = None, is_training = None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape = kernel_shape),
        dtype = tf.float32, trainable = True, name = 'kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape = [output_dims]),
            dtype = tf.float32, trainable = True, name = 'bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        if bn:
            x = batch_norm(x, is_training = is_training, bn_decay = bn_decay)
        x = activation(x)
    return x

def batch_norm(x, is_training, bn_decay):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    beta = tf.Variable(
        tf.zeros_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'beta')
    gamma = tf.Variable(
        tf.ones_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True, name = 'gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay = decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate = drop),
        lambda: x)
    return x

#######################################  ST-GCN #########################################
# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]
    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        print(x.shape, wt.shape)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def spatio_conv_layer(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels

    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)
