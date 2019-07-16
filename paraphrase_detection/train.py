import os
import tensorflow as tf
import numpy as np
import pandas as pd
from modules import *
import sklearn as sk
import sklearn.metrics
from sklearn.metrics import confusion_matrix


class Trainer():

    def build(self, max_len, vocab_lookup, embds, n_classes, num_layers, 
              filters_size, num_filters):
        """
        @max_len: Maximal length of the sentences in all training, 
                  development and test set
        @vocab_lookup: Vocabulary lookup by index
        @embds: Embeddings of all words
        @n_classes: Number of classes
        @num_layers: Number of layers
        @filters_size: List of filter size for convolution
        @num_filters: Number of filters for convolution
        @return: Output tensor of the neural network 
                 (paraphrase / not paraphrase)
        """
        # placeholders
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.x1 = tf.placeholder("string", [None, max_len], name="x1")
        self.x2 = tf.placeholder("string", [None, max_len], name="x2")
        self.y = tf.placeholder("float32", [None, n_classes], name="y")
        # embeddings for 1st sentence
        x1_embd = tf.nn.l2_normalize(get_sentences_embd(self.x1, 
                                                        vocab_lookup, 
                                                        embds), dim=1)
        x1_embd = tf.transpose(x1_embd, perm=[0, 2, 1])
        x1_embd = tf.reshape(x1_embd, [-1, x1_embd.shape[1], 
                             x1_embd.shape[2], 1])
        # embeddings for 2nd sentence
        x2_embd = tf.nn.l2_normalize(get_sentences_embd(self.x2, 
                                                        vocab_lookup, 
                                                        embds), dim=1)
        x2_embd = tf.transpose(x2_embd, perm=[0, 2, 1])
        x2_embd = tf.reshape(x2_embd, [-1, x2_embd.shape[1], 
                             x2_embd.shape[2], 1])
        # represent sentences with convolution, averaging, k-max-pooling
        with tf.variable_scope("conv_weights", reuse=tf.AUTO_REUSE):
            weights = []
            for i in range(num_layers):
                weight = []
                for j in range(num_filters[i]):
                    weight.append(tf.get_variable(
                        "weight_%d_%d" % (i+1, j+1), 
                        [1, filters_size[i][j], 1, 1],
                        #initializer=tf.initializers.random_normal()
                        initializer=tf.contrib.layers.xavier_initializer()
                        ))
                weights.append(weight)
        print("\n====Network Architecture=====\n") 
        f_l = ["sn", "ln"]
        t_pool_1, features_1 = represent_sentence(x1_embd, weights, 
                                                  filters_size, 
                                                  num_layers, 
                                                  num_filters, 
                                                  self.keep_prob, 
                                                  s_id=1, f_l=f_l)
        t_pool_2, features_2 = represent_sentence(x2_embd, weights, 
                                                  filters_size, 
                                                  num_layers, 
                                                  num_filters, 
                                                  self.keep_prob, 
                                                  s_id=2, f_l=f_l)
        # get all features and build feature matrix
        with tf.variable_scope("features", reuse=tf.AUTO_REUSE):
            print("- features:")
            if features_1 and features_2:
                f_all = []
                for i in range(len(features_1)):
                    f = calculate_feature_matrix(features_1[i], 
                                                 features_2[i])
                    print("shape of feature: {}".format(f.shape))
                    f_all.append(f)
                combined_f_all = combine_features(f_all, 64)
            else:
                combined_f_all = tf.contrib.layers.flatten(
                        tf.concat([t_pool_1, t_pool_2], axis=1))    
            print("shape of all features after combining: {}".format(
                combined_f_all.shape))       
        # flatten the feature matrix and classifiy
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
            print("- classifier:")
            output = classify(tf.contrib.layers.flatten(combined_f_all), 
                              n_classes)
            print("shape after classification: {}".format(output.shape))
        return output

    def train(self, output, data, labels, num_epochs, batch_size, 
              learning_rate):
        """
        @output: Output tensor of the neural network 
                 (paraphrase / not paraphrase)
        @data: Dictionary of embeddings of sentence pairs in all 
               three sets
        @labels: Dictonary of labels in all thress sets
        @num_epochs: Number of training epochs
        @batch_size: Batch size for mini-batch gradient descent
        @learning_rate: Learning rate
        """
        train_x = data['train']
        train_y = labels['train']
        dev_x = data['dev']
        dev_y = labels['dev']
        test_x = data['test']
        test_y = labels['test']
       
        output = tf.identity(output, name="output")
        # Define loss functions, optimizer, accuracy
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                            labels=self.y
                                                            )
                    )
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(loss)            
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(output, 1), 
                                          tf.argmax(self.y, 1))
            acc_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Initialize
        init = [tf.global_variables_initializer(), 
                tf.tables_initializer()]
        # List all variables to be trained
        vs = tf.trainable_variables()
        print("Number of trainable variables: {}".format(len(vs)))
        for var in vs:
            print(var)
        print("\n====Training=====\n")
        # Starting training process
        best_validation_accuracy = 0.0
        saver = tf.train.Saver(tf.global_variables())
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'best_validation')

        train_x1 = np.array(train_x[0])
        train_x2 = np.array(train_x[1])
        train_y = np.array(train_y)

        dev_x1 = np.array(dev_x[0])
        dev_x2 = np.array(dev_x[1])
        dev_y = np.array(dev_y)

        test_x1 = np.array(test_x[0])
        test_x2 = np.array(test_x[1])
        test_y = np.array(test_y)
        
        with tf.Session() as sess:
            sess.run(init)           
            for epoch in range(num_epochs):
                num_batches = int(train_x1.shape[0] / batch_size)        
                # shuffle training data
                arr = np.arange(len(train_y))
                np.random.shuffle(arr)
                train_x1_shuffle = train_x1[arr]
                train_x2_shuffle = train_x2[arr]
                train_y_shuffle = train_y[arr]
                for i in range(num_batches):
                    start = i*batch_size
                    end = (i+1)*batch_size
                    batch_x1 = train_x1_shuffle[start:end] 
                    batch_x2 = train_x2_shuffle[start:end]
                    batch_y = train_y_shuffle[start:end]
                    sess.run([optimizer], feed_dict={self.x1: batch_x1, 
                                                    self.x2: batch_x2, 
                                                    self.y: batch_y, 
                                                    self.keep_prob: 0.7})
                accuracy_train, loss_train = sess.run([acc_op, loss], 
                                             feed_dict={self.x1: train_x1, 
                                                        self.x2: train_x2, 
                                                        self.y: train_y,
                                                        self.keep_prob: 1.0})
                accuracy_dev, loss_dev = sess.run([acc_op, loss],
                                         feed_dict={self.x1: dev_x1, 
                                                    self.x2: dev_x2, 
                                                    self.y: dev_y, 
                                                    self.keep_prob: 1.0})
                y_pred = tf.argmax(output, 1).eval({self.x1: dev_x1, 
                                                    self.x2: dev_x2, 
                                                    self.keep_prob: 1.0})
                y_true = np.argmax(dev_y, 1)
                f1_dev = sk.metrics.f1_score(y_true, y_pred)
                print("""Epoch: %3.i - Acc_train: %.3f, Acc_dev: %.3f, 
                        Loss_train: %.3f, Loss_dev: %.3f, F1_dev: %.3f
                      """ % (epoch, accuracy_train, accuracy_dev, 
                           loss_train, loss_dev, f1_dev))
                # find out the best result of the validation set
                if accuracy_dev > best_validation_accuracy:
                    best_validation_accuracy = accuracy_dev
                    saver.save(sess=sess, save_path=save_path)                    
            print("Optimization Finished!")
            saver.restore(sess=sess, save_path=save_path)
            accuracy_test = acc_op.eval({self.x1: test_x1, 
                                         self.x2: test_x2, 
                                         self.y: test_y, 
                                         self.keep_prob: 1.0})
            y_pred = tf.argmax(output, 1).eval({self.x1: test_x1, 
                                                self.x2: test_x2, 
                                                self.keep_prob: 1.0})
            y_true = np.argmax(test_y, 1)
            f1_test = sk.metrics.f1_score(y_true, y_pred)
            print ("Acc_test: %.3f,  F1_test: %.3f" % (accuracy_test, f1_test))
            print(sk.metrics.confusion_matrix(y_true, y_pred))

