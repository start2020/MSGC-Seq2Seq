import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import model_common

def placeholder(Batch_Size, P, Q, N, F_in, F_out):
    labels = tf.compat.v1.placeholder(shape = (Batch_Size, Q, N, F_out), dtype = tf.float32,name="lables")
    samples = tf.compat.v1.placeholder(shape = (Batch_Size, P, N, F_in), dtype = tf.float32,name="samples")
    is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool, name='is_training')
    return labels, samples, is_training

# X:(B,P,N,1), adj:(N,N)
# output:(B,P,N,1)
def adjacent(args, X, La):
    #output = tf.matmul(La, X)
    units_gcn = [int(i) for i in args.units_gcn.split("-")]
    output = model_common.multi_gcn(La, X, args.activations_gcn, units_gcn, args.Ks_gcn)
    return output

# X:(B,P,N,1), adj:(Q,P,N,N)
# output:(B,P,N,Q)
def reachable_en(args, X, Lr):
    Q,P,N,_ = Lr.shape
    output = []
    for p in range(P):
        xp = X[:,p,...] #(B,N,1)
        xq = []
        for q in range(Q):
            L = Lr[q,p,...] #(N,N)
            L = tf.nn.softmax(L, axis=-1)
            units_gcn = [int(i) for i in args.units_gcn.split("-")]
            x = model_common.multi_gcn(L, xp, args.activations_gcn, units_gcn, args.Ks_gcn)
            xq.append(x)
        xq = tf.concat(xq,axis=-1) #(B,N,Q)
        output.append(xq)
    output = tf.stack(output, axis=1) #(B,P,N,Q)
    return output

def XST(dw_mat,X, TE, SE, D,T):
    SE = model_common.s_embbing_static(SE, D, activations=['relu', None]) #(N,Fs)=>(1,1,N,D)
    B,P,N,_ = X.shape
    inputs = X
    TE = model_common.t_embbing_static(dw_mat,2, TE, T, D, activations=['relu', None]) #(B,P,N,D)
    X = model_common.x_embedding(X, D, activations=['relu', None])

    is_TE = True
    is_SE = True
    X = model_common.x_SE_TE(X, SE, TE, is_X=True,is_SE=is_SE , is_TE=is_TE)
    return inputs, X

# X:(B,P,N,1),SE:(N,F), T:(B,P,N,2)
# output: (B,P,N,1)
def dynamic(args, X,D):
    # (B,P,N,2D)=> (B,P,N,D)
    query = model_common.multi_fc(X, activations=['relu'], units=[D])
    key = model_common.multi_fc(X, activations=['relu'], units=[D])
    value = model_common.multi_fc(X, activations=['relu'], units=[D])
    # (B,P,N,D)*(B,P,N,D)=>(B,P,N,N)
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (D ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # (B,P,N,N) * (B,P,N,1)=> (B,P,N,1)
    #inputs = tf.matmul(attention, value)
    units_gcn = [int(i) for i in args.units_gcn.split("-")]
    inputs = model_common.multi_gcn(attention, value, args.activations_gcn,units_gcn , args.Ks_gcn)
    return inputs

def is_train(labels, q, output):
    x = labels[:, q, ...]
    # Return either the model's prediction or the previous ground truth in training.
    Use_Curriculum_Learning = True
    Cl_Decay_Steps = 2000
    if Use_Curriculum_Learning:
        c = tf.random_uniform((), minval=0., maxval=1.)
        k = Cl_Decay_Steps * 1.0
        global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
        threshold = tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)
        x = tf.cond(tf.less(c, threshold), lambda: labels[:, q, ...], lambda: output)
    return x

def is_test(output):
    return output

'''
X: (B,P,N,3) = (B,P,N,1)[X] + (B,P,N,2)[TE]
Labels:(B,P,N,1)
La:(N,N)
Lr:(Q,P,N,N)
SE:(N,F)
'''
def Model(args, mean, std, X, labels, La, Lr, SE, is_training,dw_mat):
    TE = tf.cast(X[..., -2:], tf.int32)  # (B,P,N,2)
    X = X[..., :-2]  # (B,P,N,1)
    D, T  = args.D, args.T
    num_units = [int(i) for i in args.num_units.split("-")]
    print(num_units)
    B,Q,N,F = labels.shape #(B,Q,N,1)
    labels = tf.squeeze(tf.concat(tf.split(labels, N, axis=2),axis=0),axis=2) #(BN,Q,1,1)=>(BN,Q,1)
    inputs, X = XST(dw_mat,X, TE, SE, D, T)




    X_a = adjacent(args, X, La)  # (B,P,N,1)
    X_d = dynamic(args, X, D)  # (B,P,N,1)
    X_ad = tf.concat([X_a, X_d], axis=-1)  # (B,P,N,Q+2)

    outputs = X_ad
    samples = tf.squeeze(tf.concat(tf.split(outputs, N, axis=2), axis=0)) #(B,P,1,Q+2)=>(BN,P,1,Q+2)=>(BN,P,Q+2)

    X_r = reachable_en(args, X, Lr)  # (B,P,N,Q)
    X_r = tf.transpose(X_r, [0, 2, 1, 3])  # (B,N,P,Q)
    (B1, N1, P1, Q1) = X_r.shape
    X_r = tf.reshape(X_r, [B1 * N1, P1, Q1])  # (BN,P,Q)


    # encoder
    with tf.variable_scope("encoder"):
        multi_gru_en = model_common.multi_cells(num_units)
        # outputs:(B,T,F),last_states:tuple, [(B,F),(B,F)]
        outputs, last_states = tf.nn.dynamic_rnn(cell=multi_gru_en, inputs=samples, dtype=tf.float32)

    # decoder
    Hs = outputs  # (B,T,F)
    P = outputs.shape[1]
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        multi_gru_de = model_common.multi_cells(num_units)
        outputs = []
        inputs = tf.zeros_like(labels[:, 0, ...])
        for q in range(Q):

            X_rq = X_r[..., q]  # (BN,P)
            inputs = tf.concat([inputs, X_rq], axis=-1)  # (B,2F)

            HCq = []
            for h in range(args.H):
                # temporal attention
                Sq_1 = last_states[1]  # (B,F)
                eh = []
                for p in range(P):
                    Hp = Hs[:, p, :]  # (B,F)
                    shp = tf.concat([Hp, Sq_1], axis=-1)  # (B, 2F)
                    w = tf.get_variable('w_%d' % h, [shp.shape[-1], args.Fe], dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer())  # (2F,Fe)
                    ehp = tf.einsum('ab,bc->ac', tf.nn.tanh(shp), w)  # (B, 2F)*(2F,Fe)=>(B,Fe)
                    v = tf.get_variable('v_%d' % h, [args.Fe, 1], dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer())  # (Fe,1)
                    ep = tf.einsum('ab,bc->ac', tf.nn.tanh(ehp), v)  # (B, Fe)*(Fe,1)=>(B,1)
                    eh.append(ep)
                eh = tf.concat(eh, axis=-1)  # (B,P)
                eh = tf.nn.softmax(eh, axis=-1)  # (B,P)
                eh = tf.expand_dims(eh, axis=-1)  # (B,P,1)


                Cq = tf.zeros_like(Hs[:, 0, :])  # (B,F)
                for p in range(P):
                    ap = eh[:, p, :]  # (B,1)
                    Hp = Hs[:, p, :]  # (B,F)
                    Hap = Hp * ap  # (B,F)
                    Cq += Hap
                HCq.append(Cq)

            HCq = tf.concat(HCq, axis=-1)  # (B,HF)
            w = tf.get_variable('w', [HCq.shape[-1], Cq.shape[-1]], dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # (HF,F)
            Cq = tf.matmul(HCq, w)  # (B,HF)*(HF,F) =>(B,F)


            inputs = tf.concat([inputs, Cq], axis=-1)  # (B,2F)

            output, states = multi_gru_de(inputs, last_states)
            output = model_common.fc(output, F)
            outputs.append(output)
            last_states = states
            inputs = tf.cond(is_training, lambda: is_train(labels, q, output), lambda: is_test(output))
        outputs = tf.stack(outputs, axis=1)  # (BN,Q,F)
    outputs = tf.stack(tf.split(outputs, N, axis=0), axis=2)  # (BN,Q,F)=>(B,Q,N,F)
    outputs = model_common.inverse_positive(outputs, std, mean)
    return outputs