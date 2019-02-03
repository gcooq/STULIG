# -*- coding:  UTF-8 -*-
'''
Created on 2018.08.03
@author: liuxin
'''
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.layers.core import Dense
from attention import attention
from compiler.ast import flatten
import matplotlib.pyplot as plt
import time
import math


# paramters
# paramters
FFF = open('aoa.dat','w')
dec_in_channels = 1
batch_size = 32  # you can choose 16,or ...
iter_num = 25
n_input = 250  # embedding size
n_hidden = 300  # vae embeddings
c_hidden = 512  # classifer embedding
bata = 0.8
keep_prob = tf.placeholder("float")
alpha = tf.placeholder("float")
it_learning_rate = tf.placeholder("float")
z_size = 50
inputs_decoder = 64 * dec_in_channels / 2

# data set
label_size = 112
n_latent = 50
reshaped_dim = [-1, 8, 8, 1]
#
# tensor definition

input_x = tf.placeholder(dtype=tf.int32)
l_y = tf.placeholder(dtype=tf.int32, shape=[batch_size, label_size])
vae_y = tf.placeholder("float", [batch_size, None, label_size])  # vae_yTUL_CNN
vae_y_u = tf.placeholder("float", [label_size, batch_size, None, label_size])

target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

u_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
u_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
latentscale_iter = tf.placeholder(dtype=tf.float32)
pos_ = tf.placeholder(dtype=tf.float32)
# global list
table_X = {}  # trajectory
new_table_X = {}
new_table_X = {}
voc_tra = list()
# define the weight and bias dictionary
with tf.name_scope("weight_inital"):
    weights_de = {
        'w_': tf.Variable(tf.random_normal([z_size, n_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([2 * c_hidden, label_size]))
    }
    biases_de = {
        'b_': tf.Variable(tf.random_normal([n_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([label_size]))
    }
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x

def absolute_pos():
    sess = tf.Session()
    x = []
    for i in range(0, batch_size):
        for j in range(0, 1):
            for z in range(0, 250):
                x.append(j)
    x = tf.reshape(x, [batch_size, 1, 250])
    x = sess.run(x)
    return x


pos1_ = absolute_pos()

def extract_character_vocab(total_T):
    special_words = ['<PAD>', '<GO>', '<EOS>']
    set_words = list(set(flatten(total_T)))
    set_words = sorted(set_words)
    set_words = [str(item) for item in set_words]
    print len(set_words)
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def extract_words_vocab():
    print 'dictionary length',len(voc_tra)
    int_to_vocab={idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def getPvector(i):  # Embedding tensor
    return new_table_X[i]
def get_index(userT):
    userT = list(set(userT))
    User_List = sorted(userT)
    # print userT
    return User_List
def get_mask_index(value, User_List):
    #     print User_List #weikong
    return User_List.index(value)
def get_true_index(index, User_List):
    return User_List[index]
def getXs():  # =
    fpointvec = open('data/gowalla_user_vector250d_.dat', 'r')  # it has used word2vec
    #     table_X={}  #=
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  #
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  #
        if lineArr[0] == '</s>':
            table_X['<PAD>']=X  #dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] =X
    print "point number item=", item
    return table_X
def readtraindata():
    test_T = list()
    test_UserT = list()
    test_lens = list()
    ftraindata = open('data/total.dat',
                      'r')
    tempT = list()
    pointT = list()
    userT = list()
    seqlens = list()
    item = 0
    test = list()
    pointtt = list()
    # for line in ftraindata.readlines():
    #     lineArr = line.split()
    #     X = list()
    #     for i in lineArr:
    #         X.append(str(i))  # chanage to string or char type
    #     tempT.append(X)
    #     userT.append(int(X[0]))
    #     pointT.append(X[1:])
    #     seqlens.append(len(X) - 1)
    #     item += 1
    count = 1
    for line in ftraindata.readlines():
        line = line.replace('\r\n', '')
        lineArr = line.split(',')
        userT.append(lineArr[0])
        for i in range(1, len(lineArr)):
            if count == 1:
                test.append(lineArr[i])
                pointtt.append(lineArr[i])
                count = count + 1
            elif count == 4:
                test.append(lineArr[i])
                tempT.append(test)
                count = 1
                test = []
            else:
                test.append(lineArr[i])
                count = count + 1
        pointT.append(tempT)
        pointtt = []
        seqlens.append((len(lineArr) - 1) / 4)
        item = item + 1
        test = []
        tempT = []
    Train_Size =10000 #small data size for gowalla 112 user
    pointT = pointT[:Train_Size]  # all tra
    userT = userT[:Train_Size]  # all user
    seqlens = seqlens[:Train_Size]  # all length
    User_List = get_index(userT)
    flag = 0
    count = 0;
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  #
    rate = 0.5 #split rate
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):
            User += 1
            #split data
            if (count > 1):  #
                test_T += (pointT[int((index - math.ceil(count * rate))):index])
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1;
            flag = userT[index]
        else:
            count += 1
    pointT = temp_pointT
    userT = temp_userY
    total_T = pointT + test_T
    print 'Total Numbers=', item - 1
    print 'train trajectories number=', len(total_T)
    print 'Train Size=', len(pointT), ' Test Size=', len(test_T), "User numbers=", len(User_List)
    return pointT, userT,test_T, test_UserT,User_List#
#input
getXs()
pointT, userT,test_T, test_UserT,User_List=readtraindata()
total_Ts=pointT+test_T
for i_ in range(len(total_Ts)):
    for j_ in range(len(total_Ts[i_])):
        new_table_X[total_Ts[i_][j_][0]] = table_X[total_Ts[i_][j_][0]]
        # new_table_X_t[total_Ts[i_][j_][1]] = table_X_t[total_Ts[i_][j_][1]]
#
new_table_X['<GO>']=table_X['<GO>']
new_table_X['<EOS>']=table_X['<EOS>']
new_table_X['<PAD>']=table_X['<PAD>']
for keys in new_table_X:
    voc_tra.append(keys)
print 'train trajectory size',len(pointT)
print 'test trajectory size',len(test_T)

int_to_vocab, vocab_to_int=extract_words_vocab()
print 'POIs number is ',len(vocab_to_int)
TOTAL_SIZE = len(vocab_to_int)

#convert to int type
#Train Dataset
new_pointT = list()
for i in range(len(pointT)):
    temp = list()
    for j in range(len(pointT[i])):

        temp.append(vocab_to_int[pointT[i][j][0]])
    new_pointT.append(temp)

#Test Dataset
new_testT = list()
for i in range(len(test_T)):
    temp = list()
    for j in range(len(test_T[i])):
        temp.append(vocab_to_int[test_T[i][j][0]])
    new_testT.append(temp)
#Get dictionary
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())
print 'Dictionary Size',len(dic_em())



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

# ------------------------------------------------------------------------------
# classifer
def classifer(encoder_embed_input,keep_prob=0.5,reuse=False):
    attention_size = 50
    with tf.variable_scope("classifier",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        #encoder_input_t = tf.nn.embedding_lookup(dic_embeddings_t, encoder_embed_input_t)
        encoder_input = encoder_input + pos1_
        # encoder_input_ = tf.concat([encoder_input, encoder_input_t], 2)
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # , state_is_tuple=True
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # add dropout
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                    state_is_tuple=True)  # , state_is_tuple=True
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # add dropout
        #
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32, time_major=True)
        new_outputs = tf.concat(outputs,2)
        pred = (tf.matmul(new_outputs[-1], weights_de["out"]) + biases_de["out"])
        return pred


def conv_network(l_y,input_, output_dim, reuse, rate, out_act= None, std_bias=0, ):
    output = None
    h = dict()
    reuse = False
    # Convolutional Layer #1 output = [batch_size,28, 20, 16]
    filters = 16
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = tf.layers.conv2d(input_, filters, kernel_size, strides, padding,  reuse=reuse )

# Convolutional Layer #2 output = [batch_size,28, 20, 32]
    filters = 32
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = tf.layers.conv2d(input_, filters, kernel_size, strides, padding,  reuse=reuse )

    # # Convolutional Layer #3 output = [batch_size,14, 10, 32]
    filters = 32
    kernel_size=3
    strides=1
    padding='same'
    x = tf.layers.conv2d(x, filters, kernel_size, strides, padding,  reuse=reuse )

    # Pooling Layer #1 output = [batch_size,14, 14,32]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool1')

    # Convolutional Layer #4 output = [batch_size,7, 5, 64]
    # filters = 64
    # kernel_size=3
    # strides=1
    # padding='same'
    # x = tf.layers.conv2d(x, filters, kernel_size, strides, padding,  reuse=reuse )
    #
    # # Convolutional Layer #5 output = [batch_size,4, 3, 64]
    # filters = 64
    # kernel_size=2
    # strides=1
    # padding='same'
    # x = tf.layers.conv2d(x, filters, kernel_size, strides, padding, reuse=reuse )

    # Convolutional Layer #5 output = [batch_size,4, 3, 64]
    filters = 64
    kernel_size=2
    strides=1
    padding='same'
    x = tf.layers.conv2d(x, filters, kernel_size, strides, padding, reuse=reuse )
    print(x)

    # # Pooling Layer #2 output = [batch_size,7, 7,64]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool2')

    # Dense
    # x = tf.concat(x, l_y)
    x = tf.contrib.layers.flatten(x)
    print(x)
    # x = tf.concat(x,l_y)
    h = tf.layers.dense(inputs=x, units=output_dim, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(), name='dense_1',reuse=reuse, bias_initializer=tf.constant_initializer(0.0))
    out = tf.layers.dropout(h, rate=rate)
    if(std_bias<=0):
        output = tf.layers.dense(out, output_dim,  activation = out_act,bias_initializer=tf.constant_initializer(0.0), reuse=reuse)
    else:
        output = tf.layers.dense(out, output_dim,  activation=out_act,bias_initializer=tf.truncated_normal_initializer(stddev=std_bias), reuse=reuse)

    return output, h

def deconv(input_, filters, k_size, strides, padding, name, act_func=tf.nn.relu, kernel_init = tf.contrib.layers.variance_scaling_initializer(), bias_init = tf.constant_initializer(0.0), reuse=None ):
    deconv = tf.layers.conv2d_transpose(input_, filters,k_size, strides=strides, padding=padding, activation=act_func, kernel_initializer=kernel_init, bias_initializer=bias_init, name=name, reuse=reuse)
    print('[*] Layer (',deconv.name, ') output shape:', deconv.get_shape().as_list())

    return deconv

def deconv_network(input_, output_dim, reuse, rate, out_act= None):
    output = None
    h = dict() # [-1, aux_size, aux_size, 128]

    # Deconvolutional Layer #1 output = [batch_size,14, 14,64]
    filters = 32
    kernel_size= 2 # [aux_size +1 , aux_size +1 ]
    strides=1
    padding='same'
    x= deconv(input_, filters, kernel_size, strides, padding, 'deconv1', reuse=reuse )

    filters = 32
    kernel_size= 2
    strides=1
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv2', reuse=reuse )

    #
    # filters = 16
    # kernel_size= 3
    # strides=1
    # padding='same'
    # x= deconv(x, filters, kernel_size, strides, padding, 'deconv3', reuse=reuse )
    #
    # filters = 16
    # kernel_size=5
    # strides=1
    # padding='same'
    # x= deconv(x, filters, kernel_size, strides, padding, 'deconv4', reuse=reuse )
    #
    # # Convolutional Layer #2 output =[batch_size,28, 28,channel_num]
    filters = 16
    kernel_size=5
    strides=1
    padding='same'
    output = deconv(x, filters, kernel_size, strides, padding, 'deconv5', reuse=reuse, act_func=out_act )

    x = tf.nn.dropout(output, keep_prob)
    x = tf.contrib.layers.flatten(x)
    # print(x)
    x = tf.layers.dense(x, units=250, activation=tf.nn.sigmoid)
    # print(x)
    output = tf.reshape(x, shape=[-1, 1, 250])
    return output, h



def variable_summary(var, name='summaries'):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))

        tf.summary.histogram('histogram', var)
    return

def dense_dropout(input_, output_dim, name, rate, act_func=tf.nn.relu, kernel_init =tf.variance_scaling_initializer(), bias_init=tf.constant_initializer(0.0), reuse=None):

    h = tf.layers.dense(inputs=input_, units=output_dim, activation=act_func, kernel_initializer=kernel_init, name=name, reuse=reuse, bias_initializer=bias_init)
    out = tf.layers.dropout(h,rate=rate,name=name+'_dropout')
    print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())

    with tf.variable_scope(name, reuse=True):
        variable_summary(tf.get_variable('kernel'), 'kernel')
        variable_summary(tf.get_variable('bias'), 'bias')
    return out

def dense_network(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None):
    output = None
    h = dict()
    print("")

    # h['H1'] = tf.layers.dense(inputs=input_network, units=hidden_dim, activation=act_func, kernel_initializer=def_init, name='layer_1', reuse=reuse)
    h['H1'] = dense_dropout(input_, hidden_dim,'dense_1', rate, reuse=reuse)

    for i in range(2, num_layers + 1):
        if(i == num_layers):
            output = tf.layers.dense(h['H' + str(i - 1)], output_dim, reuse=reuse)
            # enc_mean = densei_dropout(h['H' + str(i - 1)], output_dim, None, def_init, 'layer_' + str(i), rate, reuse=reuse)
        else:
            # h['H' + str(i)] = tf.layers.dense(inputs=h['H' + str(i - 1)], units=input_dim, activation=act_func, kernel_initializer=def_init, name='layer_' + str(i), reuse=None)
            h['H' + str(i)] = dense_dropout(h['H' + str(i - 1)], hidden_dim, 'dense_' + str(i), rate, reuse=reuse)

    if(num_layers==1):
        output = h['H1']

    return output, h

def sigma(tensor):
    return tf.add(tf.nn.softplus(tensor), 0.1)
max_value = 1
max_value_var = 5
act_func_mean = None
act_func_var = tf.tanh
w_dim = 128
K_clusters = 15

# ENCODER PART of VAE
def encoder(l_y,encoder_embed_input,keep_prob=0.5,reuse=False,std_bias=0):
    with tf.variable_scope("encoder",reuse=reuse):
        # print(encoder_embed_input)
        # print(encoder_embed_input_t)
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        # for i in encoder_input:
        #     for j in range(0,len(i)-1):
        #         for k in range(0,len(j)-1):
        #             FFF.write(str(k) + ' ')
        #         FFF.write('\n')
        #encoder_input_t = tf.nn.embedding_lookup(dic_embeddings_t,encoder_embed_input_t)
        activation = lrelu
        encoder_input = encoder_input + pos1_
        # encoder_input_ = tf.concat([encoder_input, encoder_input_t], 2)
        x_ = tf.reshape(encoder_input, [-1, batch_size * n_input])
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        X = tf.reshape(input_, shape=[-1, 1,250,1 ])
        with tf.variable_scope('Qz_x_mean'):
            # Notice the bias is initialize with tf.truncated_normal
            z_mean, h_z_mean = conv_network(l_y,X, n_latent,  reuse, keep_prob,
                                                  out_act=act_func_mean, std_bias=std_bias)
            z_mean = tf.scalar_mul(max_value, z_mean)
        # VARIANCE
        with tf.variable_scope('Qz_x_var'):
            z_var_aux, h_z_var = conv_network(l_y,X, n_latent, reuse, keep_prob,
                                                    out_act=act_func_var)
            z_var_aux = tf.scalar_mul(max_value_var, z_var_aux)
            z_var = sigma(z_var_aux)

        with tf.variable_scope('Qw_x_mean'):
            w_mean, h_w_mean = conv_network(l_y, X, w_dim,  reuse, keep_prob)
        # VARIANCE
        with tf.variable_scope('Qw_x_var'):
            w_var_aux, h_w_logvar = conv_network(l_y ,X, w_dim, reuse, keep_prob,
                                                       out_act=tf.tanh)
            w_var = sigma(w_var_aux)
        return encoder_input,x_,z_mean, z_var,w_mean, w_var
        #

def Pz_wy(w, z_dim, reuse, rate,K_clusters, hidden_dim=64, num_layers=2):

    with tf.variable_scope('Pz_wy'):
        h_out, _ = dense_network(w, hidden_dim, hidden_dim, num_layers-1,reuse, rate, out_act= tf.nn.relu)

        z_means = list()
        with tf.variable_scope('mean'):
            for i in range(K_clusters):
                z_mean = dense_dropout(h_out, z_dim, 'dense_' + str(i), rate, act_func=act_func_mean,  bias_init=tf.truncated_normal_initializer(stddev=0.1), reuse=reuse)
                z_mean = tf.scalar_mul(max_value,z_mean)
                z_means.append(z_mean)
            # z_means = tf.stack(z_means)

        z_vars = list()
        with tf.variable_scope('var'):
            for i in range(K_clusters):
                z_var_aux = dense_dropout(h_out, z_dim, 'dense_' + str(i), rate, act_func=act_func_var, reuse=reuse)
                z_var_aux = tf.scalar_mul(max_value_var,z_var_aux)
                z_var = sigma(z_var_aux)
                z_vars.append(z_var)

    return z_means, z_vars


def Py_zw(z, w, z_dim, reuse, rate,K_clusters,hidden_dim=64, num_layers=2):
    with tf.variable_scope('Py_zw', reuse=reuse):
        zw = tf.concat([z, w],1, name='zw_concat')
        py_logit, h_py_logit = dense_network(zw, K_clusters, hidden_dim, num_layers,reuse, rate)
    return py_logit # [batch_size, K]


# DECODER PART of VAE
def decoder(sampled_z,keep_prob,reuse=False):
    with tf.variable_scope("decoder",reuse=reuse):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        #print(x)
        x = tf.reshape(x, reshaped_dim)
        # w = tf.constant(3, shape=(3,3, 600, 1), dtype=tf.float32, name='w')
        x_mean, _ = deconv_network(x, 2, reuse, keep_prob, out_act=tf.nn.sigmoid)

        return x_mean


def get_cost_c(pred):  # compute classifier cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=l_y))
    return cost


def get_cost_l(l_y,encoder_embed_input,reuse=False,hidden_dim = 64,K_clusters = 15,num_layers=2,z_dim = n_latent):
    input_,x_, z_mean, z_stddev, w_mean,w_stddev = encoder(l_y,encoder_embed_input, keep_prob,reuse)
    samples = tf.random_normal(tf.shape(z_stddev))
    z = z_mean + tf.exp(z_stddev * 0.5) * samples
    samples = tf.random_normal(tf.shape(w_stddev))
    w = w_mean + tf.exp(w_stddev * 0.5) * samples
    z_means, z_vars = Pz_wy(w, z_dim, reuse, keep_prob, K_clusters, hidden_dim=hidden_dim, num_layers=num_layers)

    z_vars_stack = tf.stack(z_vars)
    z_logvars_stack = tf.log(z_vars_stack)  # [K, batch_size, z_dim]
    z_means_stack = tf.stack(z_means)  # [K, batch_size, z_dim]
    samples_z = tf.random_normal(tf.shape(z_vars_stack))
    z1 = z_means_stack + tf.exp(z_vars_stack * 0.5) * samples_z
    py_logit = Py_zw(z, w, z_dim, reuse, keep_prob, K_clusters, hidden_dim=hidden_dim, num_layers=num_layers)
    py = tf.nn.softmax(py_logit)
    z_logvar = tf.log(z_stddev)
    w_logvar = tf.log(w_stddev)
    # Add small constant to avoid tf.log(0)
    log_py = tf.log(1e-10 + py)
    dec = decoder(z,keep_prob,reuse)
    #h_state = tf.nn.softplus(tf.matmul(z, weights_de['w_']) + biases_de['b_'])
    # c_state = tf.nn.softplus(tf.matmul(z, weights_de['w_2']) + biases_de['b_2'])
    # decoder_initial_state = LSTMStateTuple(h_state, encode_states[1])
    # decoder_output, predicting_logits, training_logits, masks, target = decoder(decoder_embed_input, decoder_y,
    #                                                                             target_sequence_length,
    #                                                                             max_target_sequence_length,
    #                                                                             decoder_initial_state, keep_prob, reuse)
    # KL term-------------
    unreshaped = tf.reshape(dec, [-1,  batch_size * 250])
    reconstruction = -0.5 / 1* tf.reduce_sum(tf.square(unreshaped - x_))
    loss_reconstruction_m = -tf.reduce_mean(reconstruction)
    logq = -0.5 * tf.reduce_sum(z_logvar, 1) - 0.5 * tf.reduce_sum(tf.divide(tf.square(z - z_mean), z_stddev), 1)
    z_wy = tf.expand_dims(z, 2)

    z_wy = tf.tile(z_wy, [1, 1, K_clusters])  # [batch_size, z_dim, K]
    z_wy = tf.transpose(z_wy, perm=[2, 0, 1])  # [K, batch_size, z_dim]
    log_det_sigma = tf.transpose(tf.reduce_sum(z_logvars_stack, 2))  # [batch_size, K ]

    # Shape a = tf.squared_difference(z_wy, z_means_stack): [K, batch_size, z_dim]
    # Shape b = tf.divide(a, tf.exp(z_logvars_stack)): [K, batch_size, z_dim]
    # Shape tf.reduce_sum (b, [0,2])  : [batch_size]

    aux = tf.divide(tf.square(z_wy - z_means_stack), z_vars_stack)  # [K, batch_size, z_dim]
    aux = tf.reduce_sum(aux, 2)  # [K, batch_size]
    aux = tf.transpose(aux)  # [batch_size, K]
    aux = tf.multiply(py, aux)  # [batch_size, K]
    aux = tf.reduce_sum(aux, 1)  # [batch_size]
    logp = -0.5 * tf.reduce_sum(tf.multiply(py, log_det_sigma), 1) - 0.5 * aux
    cond_prior = logq - logp
    cond_prior_m = tf.reduce_mean(cond_prior)
    KL_w = 0.5 * tf.reduce_sum(w_stddev + tf.square(w_mean) - 1 - w_logvar, 1)
    KL_w_m = tf.reduce_mean(KL_w)
    y_prior = -np.log(K_clusters, dtype='float32') - 1 / K_clusters * tf.reduce_sum(log_py, axis=1)
    y_prior_m = tf.reduce_mean(y_prior)
    # latent_loss = 0.5 * tf.reduce_mean(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
    #
    # latent_cost = tf.reduce_mean(latent_loss)
    #
    # encropy_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(unreshaped, x_), 1) ) # /batch_size

    cost = -tf.reduce_mean(reconstruction - cond_prior - KL_w - y_prior)

    return z,input_,z1,cost


def get_cost_u(u_encoder_embed_input):
    prob_y=classifer(u_encoder_embed_input,keep_prob=keep_prob,reuse=True)
    prob_y = tf.nn.softmax(prob_y)  #
    for label in range(label_size):
        y_i = get_onehot(label)
        z1,input_,z,cost_l = get_cost_l([y_i]*batch_size,u_encoder_embed_input,reuse=True)
        u_cost = tf.expand_dims([cost_l], 1)  #
        if label == 0:
            L_ulab = tf.identity(u_cost)
        else:
            L_ulab = tf.concat([L_ulab, u_cost], 1)
    U = (1. / label_size) * tf.reduce_sum(tf.multiply(L_ulab, prob_y) - tf.multiply(prob_y, tf.log(prob_y)))  #
    return U  # ,L_ulab


def creat_y_scopus(label_y, seq_length):  # copy
    lcon_y = [label_y for j in range(seq_length)]
    return lcon_y


def creat_u_y_scopus(seq_length):  #
    ucon_y = []
    for i in range(label_size):
        label_y = get_onehot(i)
        temp = []
        for j in range(batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        ucon_y.append(temp)
    return ucon_y


pred = classifer(l_encoder_embed_input)
cost_c = get_cost_c(pred)

in_z,input_,res_z,cost_l=get_cost_l(l_y,l_encoder_embed_input,reuse=False)
cost_u = get_cost_u(u_encoder_embed_input)  # unlabel data

cost = cost_c + cost_l + bata * cost_u  # alpha*

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)

# evaluate model
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(l_y, 1))  #
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]  #


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])  #
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


initial = tf.global_variables_initializer()


def train_model():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(initial)
        #saver.restore(sess, './model/gw_tulvae_112.pkt')
        print'Read train & test data'
        initial_learning_rate = 0.00008
        learning_rate_len = 0.000008
        min_kl = 0.0
        min_kl_epoch = min_kl
        kl_lens = 0.0008
        file_z = open('lan_z_25','w')
        file_z1 = open('lan_z1_25','w')
        # sort
        index_T = {}
        new_trainT = []
        new_trainT_t = []
        new_trainT_l = []
        new_trainT_w = []
        new_trainU = []
        for i in range(len(new_pointT)):
            index_T[i] = len(new_pointT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id = temp_size[i][0]
            new_trainT.append(new_pointT[id])
            # new_trainT_t.append(new_pointT_t[id])
            # new_trainT_l.append(new_pointT_l[id])
            # new_trainT_w.append(new_pointT_w[id])
            new_trainU.append(userT[id])
        # sort for test dataset
        # sort
        index_T = {}
        testT = []
        testT_t=[]
        testT_l = []
        testT_w = []
        testU = []
        for i in range(len(new_testT)):
            index_T[i] = len(new_testT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id = temp_size[i][0]
            testT.append(new_testT[id])
            # testT_t.append(new_testT_t[id])
            # testT_l.append(new_testT_l[id])
            # testT_w.append(new_testT_w[id])
            testU.append(test_UserT[id])
        # -----------------------------------
        TRAIN_ACC = []
        COST = []
        tempU = list(set(User_List))
        TRAIN_DIC = {}
        for i in range(len(tempU)):
            TRAIN_DIC[i] = [0, 0, 0]  # use mask
        TRAIN_P = []
        TRAIN_R = []
        TRAIN_F1 = []
        TRAIN_ACC1 = []
        TRAIN_ACC5 = []

        TEST_P = []
        TEST_R = []
        TEST_F1 = []
        TEST_ACC1 = []
        TEST_ACC5 = []
        Learning_rate = []
        T=[]
        count = 0
        alpha_epoch = 1
        alpha_value = (2.0 - 1.0) / iter_num
        time_s = time.time()
        for epoch in range(iter_num):

            # initial_learning_rate -= learning_rate_len
            if (initial_learning_rate <= 0):
                initial_learning_rate = 0.000001
            step = 0
            acc = 0
            acc5 = 0
            train_cost = 0
            label_cost = 0
            unlabel_cost = 0
            classifier_cost = 0
            while step < len(new_trainT) // batch_size:
                start_i = step * batch_size
                input_x = new_trainT[start_i:start_i + batch_size]
                # input_x_t = new_trainT_t[start_i:start_i + batch_size]
                # input_x_l = new_trainT_l[start_i:start_i + batch_size]
                # input_x_w = new_trainT_w[start_i:start_i + batch_size]
                input_ux = testT[start_i:start_i + batch_size]
                input_ux_t = testT_t[start_i:start_i + batch_size]
                input_ux_l = testT_l[start_i:start_i + batch_size]
                input_ux_w = testT_w[start_i:start_i + batch_size]


                #
                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                # sources_batch_t = pad_sentence_batch(input_x_t, vocab_to_int['<PAD>'])
                # sources_batch_l = pad_sentence_batch(input_x_l, vocab_to_int['<PAD>'])
                # sources_batch_w = pad_sentence_batch(input_x_w, vocab_to_int['<PAD>'])
                encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
                # encode_batch_t = eos_sentence_batch(input_x_t, vocab_to_int['<EOS>'])
                # encode_batch_l = eos_sentence_batch(input_x_l, vocab_to_int['<EOS>'])
                # encode_batch_w = eos_sentence_batch(input_x_w, vocab_to_int['<EOS>'])
                input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])

                #
                un_sources_batch = pad_sentence_batch(input_ux, vocab_to_int['<PAD>'])
                # un_sources_batch_t = pad_sentence_batch(input_ux_t, vocab_to_int['<PAD>'])
                # un_sources_batch_l = pad_sentence_batch(input_ux_l, vocab_to_int['<PAD>'])
                # un_sources_batch_w = pad_sentence_batch(input_ux_w, vocab_to_int['<PAD>'])
                un_encode_batch = eos_sentence_batch(input_ux, vocab_to_int['<EOS>'])
                # un_encode_batch_t = eos_sentence_batch(input_ux_t, vocab_to_int['<EOS>'])
                # un_encode_batch_l = eos_sentence_batch(input_ux_l, vocab_to_int['<EOS>'])
                # un_encode_batch_w = eos_sentence_batch(input_ux_w, vocab_to_int['<EOS>'])
                un_input_batch = pad_sentence_batch(un_encode_batch, vocab_to_int['<PAD>'])

                # unlabel
                un_pad_source_lengths = []
                for source in input_ux:
                    un_pad_source_lengths.append(len(source) + 1)
                # record length
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source) + 1)
                # print len(input_batch[0])
                target_maxlength = len(input_batch[0]) + 1  # get max length

                un_target_maxlength = len(un_input_batch[0]) + 1  # get max length
                if min_kl_epoch < 1.0:
                    min_kl_epoch = min_kl + count * kl_lens
                else:
                    min_kl_epoch = 1.0
                batch_y = []
                decode_y = []
                user_mask_id = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainU[y_i], User_List))
                    # print xsy_step
                    user_mask_id.append(get_mask_index(new_trainU[y_i], User_List))
                    TRAIN_DIC.get(get_mask_index(new_trainU[y_i], User_List))[2] += 1  # Groud value Groud Truth a+c
                    decode_y.append(creat_y_scopus(xsy_step, target_maxlength))  # copy
                    batch_y.append(xsy_step)
                decode_uy = creat_u_y_scopus(un_target_maxlength)
                init_z,input_en,value_z,pred_batch, c_pred, op, batch_cost, l_cost, u_cost, c_cost= sess.run(
                    [in_z,input_,res_z,pred, correct_pred, optimizer, cost, cost_l, cost_u, cost_c],
                    feed_dict={vae_y_u: decode_uy,
                               l_encoder_embed_input: sources_batch, l_y: batch_y,
                               u_encoder_embed_input: un_sources_batch,
                               it_learning_rate: initial_learning_rate, latentscale_iter: min_kl_epoch,
                               keep_prob: 0.5,alpha: alpha_epoch})
                # computing
                if (epoch == iter_num - 1):
                    for each_z in value_z:# batch
                        for value in each_z:
                            for v_z in value:# n_latent
                                file_z.write(str(v_z) + ' ')
                            file_z.write('\n')
                    for i in init_z:
                        for j in i:
                            file_z1.write(str(j) + ' ')
                        file_z1.write('\n')


                for i in range(len(pred_batch)):

                    value = pred_batch[i]

                    top1 = np.argpartition(a=-value, kth=1)[:1]
                    TRAIN_DIC.get(top1[0])[1] += 1  # recommend value a+b
                    top5 = np.argpartition(a=-value, kth=5)[:5]
                    if user_mask_id[i] in top5:
                        acc5 += 1
                    if c_pred[i] == True:
                        acc += 1
                        TRAIN_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
                # print logit.shape
                if (step % 10 == 0 and step is not 0):
                    print(step)
                    print 'min_kl_epoch', min_kl_epoch
                    print 'TRAIN LOSS', train_cost, 'LABEL COST', label_cost, 'Unlabel Cost', unlabel_cost, 'Classifier Cost', classifier_cost
                loss = np.mean(batch_cost)
                lbatch_cost = np.mean(l_cost)
                ubatch_cost = np.mean(u_cost)
                cbatch_cost = np.mean(c_cost * alpha_epoch)
                classifier_cost += cbatch_cost
                unlabel_cost += ubatch_cost
                label_cost += lbatch_cost
                train_cost += loss
                step += 1  # while
                count += 1
                time_end = time.time()
            T.append(time_end-time_s)
            alpha_epoch += alpha_value

            # Precision Recall, F1
            P = []
            R = []
            for i in TRAIN_DIC.keys():
                # print TRAIN_DIC.get(i)[0],TRAIN_DIC.get(i)[1]
                if TRAIN_DIC.get(i)[1] == 0:
                    TRAIN_DIC.get(i)[1] = 1
                if TRAIN_DIC.get(i)[2] == 0:
                    TRAIN_DIC.get(i)[2] = 1
                Pi = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[1]
                Ri = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[2]
                P.append(Pi)
                R.append(Ri)
            macro_R = np.mean(R)
            macro_P = np.mean(P)
            macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
            TRAIN_P.append(macro_P)
            TRAIN_R.append(macro_R)
            TRAIN_F1.append(macro_F1)
            TRAIN_ACC1.append(acc / (step * batch_size))
            TRAIN_ACC5.append(acc5 / (step * batch_size))
            print '\nTRAIN RESULT'
            print 'macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1
            print 'total train number', step * batch_size, 'learning rate', initial_learning_rate
            print 'iter', epoch, 'Accuracy', acc / (step * batch_size), 'Accuracy5', acc5 / (
                    step * batch_size), 'TRAIN LOSS', train_cost
            print '\nepoch TEST'
            TEST_p, TEST_r, TEST_f1, TEST_acc1, TEST_acc5 = test_model(sess, testT,testT_t,testT_l,testT_w,  testU, epoch)
            TEST_P.append(TEST_p)
            TEST_R.append(TEST_r)
            TEST_F1.append(TEST_f1)
            TEST_ACC1.append(TEST_acc1)
            TEST_ACC5.append(TEST_acc5)
            Learning_rate.append(initial_learning_rate)
            saver.save(sess, './model/gw_tulvae_112.pkt')
        save_metrics(Learning_rate, TEST_P, TEST_R, TEST_F1, TEST_ACC1, TEST_ACC5, T,root='./out/gw_tulvae_test_112.txt')
        save_metrics(Learning_rate, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, TRAIN_ACC5,T,
                     root='./out/gw_tulvae_train_112.txt')
        draw_pic_metric(TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, TRAIN_ACC5, name='train')
        draw_pic_metric(TEST_P, TEST_R, TEST_F1, TEST_ACC1, TEST_ACC5, name='test')
    file_z1.close()
    file_z.close()
        # metric_compute(correct_pred)


def test_model(sess, testT,testT_t,testT_l,testT_w, testU, epoch):
    step = 0
    count = 0
    acc = 0
    acc5 = 0
    tempU = list(set(User_List))
    TEST_DIC = {}

    for i in range(len(tempU)):
        TEST_DIC[i] = [0, 0, 0]  # use mask
    while step < len(testT) // batch_size:  #
        start_i = step * batch_size
        input_x = testT[start_i:start_i + batch_size]
        input_x_t = testT_t[start_i:start_i + batch_size]
        input_x_l = testT_l[start_i:start_i + batch_size]
        input_x_w = testT_w[start_i:start_i + batch_size]
        #
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        # sources_batch_t = pad_sentence_batch(input_x_t, vocab_to_int['<PAD>'])
        # sources_batch_l = pad_sentence_batch(input_x_l, vocab_to_int['<PAD>'])
        # sources_batch_w = pad_sentence_batch(input_x_w, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])
        #
        pad_source_lengths = []
        user_mask_id = []
        for source in input_x:
            pad_source_lengths.append(len(source) + 1)
        batch_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step = get_onehot(get_mask_index(testU[y_i], User_List))
            user_mask_id.append(get_mask_index(testU[y_i], User_List))
            TEST_DIC.get(get_mask_index(testU[y_i], User_List))[2] += 1  # Groud value Groud Truth a+c
            batch_y.append(xsy_step)
        c_pred, pred_batch = sess.run([correct_pred, pred],
                                      feed_dict={l_encoder_embed_input: sources_batch, l_y: batch_y,

                                                 keep_prob: 1.0, l_decoder_embed_input: input_batch,
                                                 target_sequence_length: pad_source_lengths})
        for i in range(len(pred_batch)):
            value = pred_batch[i]
            top1 = np.argpartition(a=-value, kth=1)[:1]
            TEST_DIC.get(top1[0])[1] += 1  # recommend value a+b
            top5 = np.argpartition(a=-value, kth=5)[:5]
            if user_mask_id[i] in top5:
                acc5 += 1
            if c_pred[i] == True:
                acc += 1
                TEST_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
        step += 1  # while
    # Precision Recall, F1
    P = []
    R = []
    for i in TEST_DIC.keys():
        if TEST_DIC.get(i)[1] == 0:
            TEST_DIC.get(i)[1] = 1
        if TEST_DIC.get(i)[2] == 0:
            TEST_DIC.get(i)[2] = 1
        Pi = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[1]
        Ri = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[2]
        P.append(Pi)
        R.append(Ri)
    macro_R = np.mean(R)
    macro_P = np.mean(P)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    print 'macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1
    print 'iter', epoch, 'Accuracy For TEST', acc / (step * batch_size), 'Accuracy5 For TEST', acc5 / (
            step * batch_size), 'total test number', step * batch_size
    return macro_P, macro_R, macro_F1, acc / (step * batch_size), acc5 / (step * batch_size)


def save_metrics(LEARN_RATE, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, TRAIN_ACC5,T, root='out/gw_metric_tulvae.txt'):
    files = open(root, 'w')
    files.write('epoch \t learning_rate \t Precision \t Recall \t F1 \t ACC1 \t ACC5\n')
    for i in range(len(TRAIN_P)):
        files.write(str(i) + '\t')
        files.write(str(LEARN_RATE[i]) + '\t')
        files.write(
            str(TRAIN_P[i]) + '\t' + str(TRAIN_R[i]) + '\t' + str(TRAIN_F1[i]) + '\t' + str(TRAIN_ACC1[i]) +'\t'+ str(
                TRAIN_ACC5[i]) +str(T[i])+ '\n')
    files.close()


def draw_pic_metric(TEST_P, Test_R, Test_F1, Test_ACC1, TEST_ACC5, name='train'):
    font = {'family': name,
            'weight': 'bold',
            'size': 18
            }
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(TEST_P) + 1, 1))
    plt.plot(train_axis, np.array(TEST_P), "b--", label="Test P")
    train_axis = np.array(range(1, len(Test_R) + 1, 1))
    plt.plot(train_axis, np.array(Test_R), "r--", label="Test R")
    train_axis = np.array(range(1, len(Test_F1) + 1, 1))
    plt.plot(train_axis, np.array(Test_F1), "g--", label="Test F1-score")
    train_axis = np.array(range(1, len(Test_ACC1) + 1, 1))
    plt.plot(train_axis, np.array(Test_ACC1), "y--", label="Test ACC1")
    train_axis = np.array(range(1, len(TEST_ACC5) + 1, 1))
    plt.plot(train_axis, np.array(TEST_ACC5), "c--", label="Test ACC5")
    plt.title(name)
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('value')
    plt.xlabel('Training iteration')
    plt.show()


if __name__ == "__main__":
    time_start = time.time()
    train_model()
    time_end = time.time()
    print('totally cost', time_end - time_start)
    print 'Model END'
