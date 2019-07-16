import os
import tensorflow as tf
from paraphrase_detection.preprocessing import tokenize
from paraphrase_detection.utils import pad_sentences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


max_len = 34

print("Please input sentence 1")
sent_1 = input()
print("Please input sentence 2")
sent_2 = input()

test = pad_sentences([[tokenize(sent_1)], [tokenize(sent_2)]], max_len)
test_x1 = test[0]
test_x2 = test[1]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoints/best_validation.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    sess.run(tf.tables_initializer())
    output = sess.graph.get_tensor_by_name('output:0')
    x1 = sess.graph.get_tensor_by_name('x1:0')
    x2 = sess.graph.get_tensor_by_name('x2:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    pred = sess.run(tf.argmax(output, 1), feed_dict={x1: test_x1,
                                                     x2: test_x2,
                                                     keep_prob: 1.0})
    if pred[0] == 1:
        print("=> paraphrase")
    else:
        print("=> not paraphrase")

