import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def reduce_std(x):
    m = tf.reduce_mean(x, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.sqrt(tf.reduce_mean(devs_squared))


def build_embeddings(vocab, words_in_word2vec, embds_in_word2vec, 
                     train_unknown_words, dev_and_test_unknown_words, 
                     num_dimensions=300):
    # build vocab lookup
    vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(vocab), default_value=0
                )
    mean = tf.reduce_mean(np.asarray(embds_in_word2vec))
    std = reduce_std(np.asarray(embds_in_word2vec))
    # embeddings for words representable by word2vec
    in_word2vec_embd = tf.get_variable(
            name="in_word2vec_embd", 
            shape=[len(words_in_word2vec), num_dimensions],
            initializer=tf.constant_initializer(
                np.asarray(embds_in_word2vec), 
                dtype=tf.float32
            ),
            trainable=False
    )
    # embeddings for unknown words in training set
    train_unknown_embd = tf.get_variable(
            name="train_unknown_embd",
            shape=[len(train_unknown_words), num_dimensions],
            initializer=tf.initializers.random_normal(
                mean=mean,
                stddev=std,
                dtype=tf.float32
            ),
            trainable=True
    )
    # embeddings for unknown words in dev and test set
    dev_and_test_unknown_embd = tf.get_variable(
            name="dev_and_test_unknown_embd",
            shape=[len(dev_and_test_unknown_words), num_dimensions],
            initializer=tf.initializers.random_normal(
                mean=mean,
                stddev=std,
                dtype=tf.float32
            ),
            trainable=False
    )
    # embeddings for the word used in padding
    padded_embd = tf.get_variable(
            name="padded_embd",
            shape=[1, num_dimensions],
            initializer=tf.zeros_initializer(),
            trainable=False
    )
    # combine all embeddings
    embds = tf.concat([in_word2vec_embd, train_unknown_embd,
            dev_and_test_unknown_embd, padded_embd], axis=0)
    return vocab_lookup, embds


def get_sentences_embd(sentences, vocab_lookup, embds):
    """
    Convert sentences to their embeddings
    @sentences: Sentences in words
    @return: Embeddings of sentences
    """
    index = vocab_lookup.lookup(sentences)
    embd = tf.nn.embedding_lookup(embds, index)
    print("shape after embedding: {}".format(embd.shape))
    return embd

def convolute(t, weight, filter_size, layer, wide=False):
    """
    Wide convolution layer for CNN, in which the elements of at the 
    margin of a matrix are mirrored and convoluted, therefore the width 
    of the resulting sequence is extended
    @t: Input tensor of the convolution layer
    @weight: Weight for convolution
    @filter_size: Filter size for wide convolution
    @layer: Id of the layer
    @return: Output of the convolution layer
    """
    with tf.variable_scope("layer_%d" % layer, reuse=tf.AUTO_REUSE):   
        conv = tf.nn.conv2d(t, weight, strides=[1, 1, 1, 1], 
                            padding='VALID')
        if wide:
            for k in reversed(range(1, filter_size)):
                head = tf.nn.conv2d(t[:, :, :k], weight[:, -k:], 
                                    strides=[1, 1, 1, 1], 
                                    padding='VALID')
                tail = tf.nn.conv2d(t[:, :, -k:], weight[:, :k], 
                                    strides=[1, 1, 1, 1], 
                                    padding='VALID')
                conv = tf.concat([head, conv, tail], axis=2)
        return conv

def average(t, layer):
    """
    Averaging layer in which each odd row and the even row behind 
    are averaged
    @t: Input tensor for the averaging layer
    @layer: Id of the layer
    @return: Output of the averaging layer
    """
    with tf.variable_scope("layer_%d" % layer, reuse=tf.AUTO_REUSE):
        num_dimension = t.shape[1]
        odd = t[:, 0::2]
        even = t[:, 1::2]
        # print(odd[:, -1].shape)
        if num_dimension % 2 != 0:
            return tf.concat([(odd[:, :-1] + even)/2., 
                tf.reshape(odd[:, -1], [-1, 1, t.shape[2], 1])], axis=1)
        return (odd + even)/2.

def _left_part(sh, l, height, width):
    arr = np.zeros([height, width-1])
    for i in range(len(sh)-1):
        out_rep_count = np.prod(sh[:i])
        inn_rep_count = l / out_rep_count / sh[i]
        tmp = np.repeat(range(sh[i]), inn_rep_count)
        arr[:, i] = np.tile(tmp, out_rep_count)
    return arr

def get_absolute_top_k(t, k):
    """
    Get k items whose absolute values are maximal
    @t: Input tensor
    @k: Number of selected items
    @return: The selected kS items
    """
    abs_t = tf.abs(t)
    t_val, t_idx = tf.nn.top_k(abs_t, k=k, sorted=False)
    t_idx_shape = t_idx.shape.as_list()
    t_idx_vol = np.prod(np.array(t_idx_shape))
    t_idx_shape_len = len(t_idx_shape)
    t_idx_sort = tf.reshape(tf.contrib.framework.sort(t_idx), t_idx_shape)
    t_idx_flat = tf.reshape(t_idx_sort, [-1, 1])
    left = _left_part(t_idx_shape, t_idx_flat.shape.as_list()[0], 
                      t_idx_vol, t_idx_shape_len) 
    new_idx = tf.concat([left, t_idx_flat], axis=1)
    res = tf.gather_nd(t, new_idx)
    res = tf.reshape(res, t.shape.as_list()[:-1] + [k])
    return res

def k_max_pool(t, layer, dynamic=True):
    """
    K-max-pooling layer in which k largest elements are selected
    @t: Input tensor for the k-max-pooling layer
    @layer: Id of the layer
    @dynamic: Whether determining k dynamically or not
    @return: Output of the k-max-pooling layer
    """
    with tf.variable_scope("layer_%d" % layer, reuse=tf.AUTO_REUSE):
        t_shape = t.get_shape().as_list()
        num_dimension = t_shape[1]
        sentence_length = t_shape[2]
        k_top = 4
        if sentence_length < k_top:
            k_dy = sentence_length
        else:
            if dynamic: 
                k_dy = max(k_top, int(sentence_length/2) + 1)
            else:
                k_dy = k_top

        t = tf.reshape(t, [-1, num_dimension, sentence_length])
        function_to_map = lambda x: get_absolute_top_k(x, k=k_dy)
        values = tf.map_fn(function_to_map, t)

        return tf.reshape(values, [-1, num_dimension, k_dy, 1]) 


def block(t, weight, filter_size, layer, keep_prob, top=False): 
    """
    A layer containing convolution, averaging and k-max-pooling
    @t: Input tensor (Embeddings of sentence pairs)
    @weight: List of weights for convolution
    @filter_size: filter size for convolution
    @layer: Layer id
    @keep_prob: 1 - dropout
    @top: Whether this is the top layer or not
    @return Output tensor of the max pooling layer
    """ 
    print("-- layer %d" % layer)
    pool_shape = t.get_shape()
    t_conv = convolute(t, weight, filter_size, layer, wide=True)
    print("shape after convolution: {}".format(t_conv.shape))
    # add dropout
    drop_out = tf.nn.dropout(t_conv, keep_prob)
    t_avg = average(drop_out, layer)
    print("shape after averaging: {}".format(t_avg.shape))
    t_pool = k_max_pool(t_avg, layer, dynamic=not top)
    print("shape after pooling: {}".format(t_pool.shape))
    return t_pool

# Param f_l: on which levels features are extracted
# It could be any subset of ["u", "sn", "ln", "s"]
# u:  unigram level      -> input
# sn: short ngram level   -> first block after averaging
# mn: middle ngram level  -> middle blocks after averaging
# ln: long ngram level    -> top block after averaging
# s:  sentence level     -> top block after pooling
# e.g.: ["sn", "ln"] means extracting features on short ngram and long ngram level

def represent_sentence(t, weights, filters_size, num_layers, num_filters, keep_prob, s_id, f_l=[]):
    """
    Represent sentences with Bi-CNN
    @t: Input tensor (Embeddings of sentence pairs)
    @weights: List of weights for convolution
    @filters_size: List of filter size of each layer
    @num_layers: Number of layers
    @num_filters: Number of filters
    @keep_prob: 1 - dropout
    @s_id: Sentence id
    @f_l: List of features
    @return: t_pool: Output tensor of the last max pooling layer
             features: List of features 
    """
    if len(weights) != num_layers or len(filters_size) != num_layers or \
            len(num_filters) != num_layers:
        return -1
    with tf.variable_scope("sentence_%d" % s_id, reuse=tf.AUTO_REUSE):
        print("- sentence %d" % s_id)
        features = []  # list to save features in each level
        if "u" in f_l: features.append(t)
         # first block
        for j in range(num_filters[0]):           
            t_pool = block(t, weights[0][j], filters_size[0][j], 1, 
                           keep_prob, top=False) 
            if "sn" in f_l: features.append(t_pool)
        # middle blocks
        for i in range(1, num_layers-1):
            t = t_pool
            for j in range(num_filters[i]):
                t_pool = block(t, weights[i][j], filters_size[i][j], 
                               i+1, keep_prob, top=False)
                if "ln%d" % i in f_l: features.append(t_pool)
        # top block
        t = t_pool
        for j in range(num_filters[num_layers-1]):
            t_pool = block(t, weights[num_layers-1][j], 
                           filters_size[num_layers-1][j], 
                           num_layers, keep_prob, top=True)
            if "s" in f_l: features.append(t_pool)
        return t_pool, features


def calculate_feature_matrix(t1, t2):
    """
    Calculate feature matrix based on the outputs of the max-pooling layer
    @t1: Output tensor of max-pooling layer of sentence 1
    @t2: Output tensor of max-pooling layer of sentence 2 
    @return: Feature matrix
    """
    # small epsilon to add before calculating norm 
    # (prevent NaN from sqrt of 0)
    epsilon = 1e-12
    num_dimension = t1.shape[1]
    f_size = t1.shape[2]
    in_tensor_1 = tf.reshape(t1, [-1, num_dimension, f_size])
    in_tensor_2 = tf.reshape(t2, [-1, num_dimension, f_size])    
    x1_expanded = tf.tile(tf.expand_dims(in_tensor_1, 1), [1, f_size, 1, 1])
    x2_expanded = tf.transpose(tf.tile(tf.expand_dims(
            tf.transpose(in_tensor_2, 
            perm=[0,2,1]), 2), [1, 1, f_size, 1]), perm=[0,1,3,2])    
    # calculate euclidean distance:
    output_features = tf.subtract(x1_expanded, x2_expanded)
    output_features = tf.add(output_features, epsilon)
    output_features = tf.norm(output_features, ord='euclidean', axis=2)
    # calculate features:
    output_features = tf.exp(tf.negative(tf.divide(output_features, 4.0)))   
    return output_features


def combine_features(f_all, r):
    """
    Combine feature matrices of each layer
    @f_all: List of features
    @r: Convert all feature matrices to r*r
    @return: Combined features matrices 
    """
    dim_l = []    # records dimension of feature matrices
    dim_l_w = []
    for f in f_all:
        dim = f.get_shape().as_list()[1]
        if dim not in dim_l:
            dim_l.append(dim)
        dim_l_w.append(dim_l.index(dim))
    # multiply feature matrices with weights
    with tf.variable_scope("dim_weights", reuse=tf.AUTO_REUSE):
        combined_f_all = []
        dim_weights = []
        for dim in dim_l:
            dim_weights.append(tf.get_variable(
                "dim_weight_%dX%d" % (dim, r), 
                [dim, r], initializer=tf.initializers.random_normal()))
        for f in f_all:
            w = dim_weights[dim_l_w[f_all.index(f)]]
            function_to_map = lambda x: tf.matmul(x, w)
            temp = tf.map_fn(function_to_map, f)
            temp2 = tf.transpose(temp, perm=[0, 2, 1])
            temp3 = tf.map_fn(function_to_map, temp2)
            combined_f_all.append(temp3)
        return tf.concat(combined_f_all, axis=1)   

def classify(features, n_classes):
    """
    Logistic regression layer
    @features: Flattened features used for classification
    @n_classes: Number of classes
    @return: Cateogrial output (paraphrase / not paraphrase)
    """
    return tf.contrib.layers.fully_connected(
            features, n_classes, activation_fn=None, 
            #weights_initializer=tf.initializers.random_normal(), 
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
            #biases_initializer=tf.initializers.random_normal()
            biases_initializer=tf.contrib.layers.xavier_initializer()
            )   



