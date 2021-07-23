import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn import linear_model
from libs import utils, metrics, data_common, graph_common
import time
import sys

def main(args):
    for i in range(args.Times):
        # 产生数据
        if args.data:
            data_common.data_process(args)
        # 实验
        if args.experiment:
            experiment(args)
            tf.reset_default_graph()
    # 是否测试
    if args.test:
        test_model(args)


def test_model(args):
    log_dir, save_dir, data_dir = utils.create_dir_pattern(args)
    data_log, train_log, result_log = utils.create_logs(args, log_dir)

    utils.log_string(train_log, str(args)[10: -1])  # 打印超参数
    model_file = save_dir + '/'

    data, _, _ = data_common.load_data(args, train_log, data_dir)
    tmp_list = []
    for i in range(len(data[2])):
        tmp = data[2]['arr_%s' % i]
        shapes = tmp.shape
        tmp_list.append(np.reshape(tmp, newshape=[shapes[0] * shapes[1], shapes[2], shapes[3], shapes[4]]))
    data = tmp_list  # 将测试集取出来
    utils.log_string(train_log, 'Data Loaded Finish...')

    # 配置GPU
    sess = GPU(number=args.GPU)

    ckpt = tf.train.get_checkpoint_state(model_file)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver2 = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        saver2.restore(sess, tf.train.latest_checkpoint(model_file))
    else:
        raise ValueError

    graph = tf.get_default_graph()  # 获取当前默认计算图


    if args.model == "RMASTGRU":
        samples = graph.get_tensor_by_name("samples:0")
        labels = graph.get_tensor_by_name("lables:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        placeholders = [labels, samples, is_training]


    preds = graph.get_tensor_by_name("preds:0")

    preds_all = []
    test_batch = int(args.Batch_Size)  # 要求大于等于2

    M = data[0].shape[0]

    # print("M==",M)
    if args.model in ["RMASTGRU"]:
        for data_index in range(int((M - 1) / test_batch)):
            feed_dict = {}
            for i in range(len(placeholders)):  # 生成 (1,N,N)的数据,解决OOM问题.为了加快速度可以增大,但是需要考虑对齐
                if i == len(placeholders) - 1:
                    feed_dict[placeholders[i]] = False
                else:
                    feed_dict[placeholders[i]] = data[i][data_index * test_batch:(data_index + 1) * test_batch, ...]

            tmp = sess.run(preds, feed_dict=feed_dict)
            if data_index == 0:
                preds_all = tmp
            else:
                preds_all = np.concatenate((preds_all, tmp), axis=0)
        preds_all = np.squeeze(preds_all,axis=-1)




    if args.model in ["RMASTGRU"]:
        labels_all = np.transpose(np.squeeze(data[0],axis=-1),[0,2,1])
        labels_all = labels_all[:preds_all.shape[0]] # 只截取一部分,用于计算指标
        preds_all = np.transpose(preds_all,[0,2,1])


    if args.round:
        preds_all = np.round(preds_all).astype(np.float32)


    mae, rmse, mape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
    Message = "MAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f" % (mae, rmse, mape)
    utils.log_string(train_log, Message)

    utils.log_string(train_log, 'Finish\n')

    utils.log_string(result_log, Message)
    sess.close()



def choose_model(args):
    if args.model == "RMASTGRU":
        import models.RMASTGRU as model
    return  model

def experiment(args):
    model = choose_model(args) # 选择模型
    log_dir, save_dir, data_dir = utils.create_dir_pattern(args)
    data_log, train_log, result_log = utils.create_logs(args, log_dir)
    utils.log_string(train_log, str(args)[10: -1])  # 打印超参数
    model_file = save_dir + '/'
    # 装载数据
    data, mean, std = data_common.load_data(args, train_log, data_dir)
    utils.log_string(train_log, 'Data Loaded Finish...')

    # 模型编译
    utils.log_string(train_log, 'compiling model...')

    if args.model == "RMASTGRU":
        # X: (V,B,P,N,1), Y: (V,B,Q,N,1)
        P, Q, N= data[0]['arr_1'].shape[-3], data[0]['arr_0'].shape[-3], data[0]['arr_0'].shape[-2]
        F_in, F_out = data[0]['arr_1'].shape[-1],data[0]['arr_0'].shape[-1]
        placeholders = model.placeholder(args.Batch_Size, P, Q, N, F_in, F_out)
        labels, samples, is_training = placeholders
        path = os.path.join(args.dataset_dir, 'originals', args.SE_file)
        SE = np.load(path)["SE"].astype(np.float32) #(N,F)
        path = os.path.join(args.dataset_dir, 'originals', args.dw_file)
        dw_mat = np.load(path)["SE"].astype(np.float32)  # (N,F)
        La = np.load(os.path.join(data_dir, args.La))['arr_0'] #(N,N)
        Lr = np.load(os.path.join(data_dir, args.Lr))['arr_0'] #(Q,P,N,N)
        preds = model.Model(args, mean, std, samples, labels, La, Lr, SE, is_training,dw_mat)

    preds = tf.identity(preds, name='preds')

    loss = metrics.masked_mae_tf(preds, labels, null_val=args.Null_Val)  # 损失
    lr, new_lr, lr_update, train_op = optimization(args, loss) # 优化
    utils.print_parameters(train_log) # 打印参数
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1) # 保存模型

    # 配置GPU
    sess = GPU(args.GPU)
    if args.continue_train:
        val_loss_min, Epoch = restore(train_log, sess, model_file, saver)
    else:
        # 初始化模型
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())

    utils.log_string(train_log, "initializer successfully")

    utils.log_string(train_log, '**** training model ****')
    wait = 0
    step = 0
    Message = ''


    epoch = Epoch
    save_loss = [[], [], []]
    while (epoch < args.max_epoch):
        # 降低学习率
        if wait >= args.patience:
            val_loss_min, epoch = restore(train_log, sess, model_file, saver)
            step += 1
            wait = 0
            New_Lr = max(args.min_learning_rate, args.base_lr * (args.lr_decay_ratio ** step))
            sess.run(lr_update, feed_dict={new_lr: New_Lr})
            if epoch > args.patience:
                for k in range(len(save_loss)):
                    save_loss[k] = save_loss[k][:-args.patience]
            if step > args.steps:
                utils.log_string(train_log, 'early stop at epoch: %04d' % (epoch))
                break

        # 打印当前时间/训练轮数/lr
        utils.log_string(train_log,
            '%s | epoch: %04d/%d, lr: %.4f' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.max_epoch, sess.run(lr)))

        # 计算训练集/验证集/测试集损失
        types = ['train', 'val', 'test']
        results = []
        for i in range(len(types)):
            feed_dicts = get_feed_dicts(args,types[i], data[i], placeholders)
            result,message = caculation(args, train_log, sess, data_dir, feed_dicts, labels, preds, train_op, loss, type=types[i])
            if i==2: # 获取测试集的性能指标
                message_cur = message
            results.append(result)


        message = "loss=> train:%.4f val:%.4f test:%.4f time=> train:%.1f val:%.1f test:%.1f" % (results[0][0], results[1][0], results[2][0], results[0][1], results[1][1], results[2][1])
        utils.log_string(train_log, message)

        # 存储损失
        save_loss[0].append(results[0][0])
        save_loss[1].append(results[1][0])
        save_loss[2].append(results[2][0])


        # 更新最小损失
        val_loss = results[1][0]
        wait, val_loss_min, Message = update_val_loss(train_log, sess, saver, model_file, epoch, wait, val_loss, val_loss_min, message_cur, Message)

        epoch += 1

    # 存储好损失
    path = os.path.join(data_dir, 'losses.npz')
    np.savez_compressed(path, np.array(save_loss))
    utils.log_string(result_log, Message)
    utils.log_string(train_log, Message)
    sess.close()

############################################# main ######################################################
def caculation(args, log, sess, data_dir, feed_dicts, labels, preds, train_op, loss, type="train"):
    start = time.time()
    loss_all = []
    preds_all = []
    labels_all = []
    message_res = ''
    for feed_dict in feed_dicts:
        if type=="train":
            sess.run([train_op], feed_dict=feed_dict)
            #loss_list = sess.run([Losses], feed_dict=feed_dict)
        batch_loss = sess.run([loss], feed_dict=feed_dict)
        loss_all.append(batch_loss)
        if type == "test":
            batch_labels, batch_preds = sess.run([labels, preds], feed_dict=feed_dict)
            preds_all.append(batch_preds)
            labels_all.append(batch_labels)
    loss_mean = np.mean(loss_all)
    Time = time.time() - start

    if type == "test":
        preds_all = np.stack(preds_all, axis=0)
        if args.round:
            preds_all = np.round(preds_all).astype(np.float32)

        labels_all = np.stack(labels_all, axis=0)
        mae, rmse, mape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
        message = "Test=> MAE:{:.4f} RMSE:{:.4f} MAPE:{:.4f}".format(mae, rmse, mape)
        message_res = "MAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f" % (mae, rmse, mape)
        utils.log_string(log, message)
        # 查看预测效果
        #pred_label(log, preds_all, labels_all, data_dir)
        # sys.exit()
    return [loss_mean, Time], message_res

def optimization(args,loss):
    lr = tf.Variable(tf.constant_initializer(args.base_lr)(shape=[]),
                    dtype = tf.float32, trainable = False, name='learning_rate') #(F, F1)
    #lr = tf.get_variable('learning_rate', initializer=tf.constant(args.base_lr), trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
    lr_update = tf.assign(lr, new_lr)
    if args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
    elif args.opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif args.opt == 'amsgrad':
        optimizer = tf.train.AMSGrad(lr, epsilon=1e-3)

    # clip
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
    return lr, new_lr, lr_update, train_op

def GPU(number):
    # GPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(number)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def restore(log, sess, model_file, saver):
    ckpt = tf.train.get_checkpoint_state(model_file)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        Epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
        val_loss_min = np.load(model_file + 'val_loss_min.npz')['loss']
        message = "restore successfully, path:%s, Epoch:%d" % (ckpt.model_checkpoint_path, Epoch)
        utils.log_string(log, message)
    else:
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())
        utils.log_string(log, "initializer successfully")
    return val_loss_min, Epoch

def get_feed_dicts(args, type, data, placeholders):
    num_batch = data['arr_0'].shape[0]
    feed_dicts = []
    if args.train_shuffle:
        per = list(np.random.permutation(num_batch))  # 随机划分
    else:
        per = range(num_batch)

    for j in per:
        feed_dict = {}
        for i in range(len(placeholders)):
            if args.model in ["GMAN","DCRNN","RMASTGRU"]:
                if i == len(placeholders) - 1:
                    if type == "train":
                        feed_dict[placeholders[i]] = True
                    else:
                        feed_dict[placeholders[i]] = False
                else:
                    feed_dict[placeholders[i]] = data['arr_%s' % i][j, ...]
            else:
                feed_dict[placeholders[i]] = data['arr_%s' % i][j, ...]
        feed_dicts.append(feed_dict)
    return feed_dicts

def update_val_loss(log, sess, saver, model_file, epoch, wait, loss, val_loss_min, message_cur, Message):
    # choose best test_loss
    if loss < val_loss_min:
        wait = 0
        val_loss_min = loss
        saver.save(sess, model_file, epoch)
        Message = message_cur
        np.savez(model_file + 'val_loss_min.npz', loss=val_loss_min)
        utils.log_string(log, "save %02d"%epoch)
    else:
        wait += 1
    return wait, val_loss_min, Message