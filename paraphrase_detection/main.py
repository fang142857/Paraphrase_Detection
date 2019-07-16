import argparse
import itertools
import numpy as np
import tensorflow as tf
from dataloader import encode_label, load_data
from utils import pad_sentences, transform, draw
from embedding import Embeddings
from gensim.models import KeyedVectors
from modules import build_embeddings
from train import Trainer


def load_datasets():
    """
    Load training, devlopment and test set
    @return: total_vocab: Total vocabulary in all three sets,
             vocabs: Dictionary of vocabs of the three sets
             data: Dictionary of sentence pairs of the three sets
             labels: Dictionary of labels of the three sets
             max_len: Maximal length of the sentences in all three sets
    """
    print("\n====Statistics about data=====\n")
    sets = ['train', 'dev', 'test']
    vocabs = {}
    data = {}
    labels = {}
    max_lens = {}
    min_lens = {}
    for s in sets:
        file_path = "./corpus/tokenized_msr_paraphrase_"+s+".txt"
        print(s+" set:")
        vocab, X, y, max_len, min_len = load_data(file_path)
        vocabs[s] = vocab
        data[s] = X
        labels[s] = y
        max_lens[s] = max_len
        min_lens[s] = min_len 
    total_vocab = list(set(itertools.chain.from_iterable(
        [vocabs[s] for s in sets])))
    print("vocab in total: {}".format(len(total_vocab)))
    max_len = max([max_lens[s] for s in sets])
    print("max length in the total dataset: {}".format(max_len))
    min_len = min([min_lens[s] for s in sets])
    print("min length in the total dataset: {}".format(min_len)) 
    return total_vocab, vocabs, data, labels, max_len  

def padding(data, max_len):
    """
    Pad sentences to maximal length
    @data: Sentence pairs
    @max_len: Maximal length of the sentences in all three sets
    @return: Padded sentence pairs
    """ 
    padded_data = {}
    for s in data.keys():
        padded_data[s] = pad_sentences(data[s], max_len)
    return padded_data

def word_embeddings(total_vocab, vocabs):  
    """
    Create word embeddings with pretrained word2vec model
    @total_vocab: Total vocabulary in all three sets
    @vocabs: Dictionary of vocabs of the three sets
    @return: vocab: Vocabulary including words in word2vec, unknown words 
                    and a word for padding
             words_in_word2vec: Words representable by word2vec
             embds_in_word2vec: Embeddings of the words representable by 
                                word2vec
             train_unknown_words: Words in training set that are not 
                                  representable by word2vec
             dev_and_test_unknown_words: Words in development and test 
                                         set that are not representable 
                                         by word2vec 
    """      
    model_path = './embeddings/GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("\n====Statistics about embeddings=====\n") 
    embedding = Embeddings(model, total_vocab)   
    words_in_word2vec = embedding.get_words_in_word2vec()
    embds_in_word2vec = embedding.get_embds_in_word2vec()
    unknown_words = embedding.get_unknown_words()
    print("words representable by Google word2vec: {}\n".format(
        len(words_in_word2vec)))
    print("unknown words: {}".format(len(unknown_words)))
    # Split unknown words into those from training and those from dev+test
    # (from training -> trainable, from dev+test -> not trainable)   
    train_vocab = vocabs['train']
    train_unknown_words = list(set(train_vocab) - set(words_in_word2vec))  
    print("\ttrain unknown words: {}".format(len(train_unknown_words)))
    dev_and_test_unknown_words = list(
            set(unknown_words) - set(train_unknown_words))
    print("\tdev and test unknown words: {}\n".format(
        len(dev_and_test_unknown_words)))
    vocab = words_in_word2vec + train_unknown_words + \
            dev_and_test_unknown_words + [""]
    print("total vocab: {}\n".format(len(vocab)))
    return vocab, words_in_word2vec, embds_in_word2vec, \
            train_unknown_words, dev_and_test_unknown_words

def vis_embds(embds, words_in_word2vec, train_unknown_words, 
              dev_and_test_unknown_words, filename):
    transformed = transform(words_in_word2vec, train_unknown_words, 
                            dev_and_test_unknown_words, embds)
    draw(transformed, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Arguments for Paraphrase Detection')
    parser.add_argument('--n_epochs', default=100, 
                        type=int, help="Number of epochs")
    parser.add_argument('--bs', default=20, type=int, help="batch size")
    parser.add_argument('--lr', default=0.001, type=float, 
                        help="learning rate")
    parser.add_argument('--n_classes', default=2, type=int, 
                        help="Number of classes")
    parser.add_argument('--n_layers', default= 3, type=int, 
                        help="Number of layers")
    parser.add_argument('--n_filters', nargs='+', type=int, 
                        default=[2, 5, 1], 
                        help="Number of filters, e.g. --n_filters 2 5 1")
    parser.add_argument('--fs', nargs='+', type=int, action='append', 
                        default=[[2, 2], [2, 2, 2, 3, 3], [3]], 
                        help="Filter size of each layer, \
                        e.g. --fs 2 2 --fs 2 2 2 3 3 --fs 3")
    parser.add_argument('--vis_embds', type=int, default=0, 
                        choices=[0, 1], 
                        help="Whether to visualize embeddings")
    tf.reset_default_graph()
    args = parser.parse_args()
    total_vocab, vocabs, data, labels, max_len  = load_datasets()
    padded_data = padding(data, max_len)
    vocab, words_in_word2vec, embds_in_word2vec, train_unknown_words, \
            dev_and_test_unknown_words = word_embeddings(total_vocab, 
                                                         vocabs)
    vocab_lookup, embds = build_embeddings(vocab, words_in_word2vec, 
                                           embds_in_word2vec, 
                                           train_unknown_words, 
                                           dev_and_test_unknown_words)
    if args.vis_embds != 0:
        filename = "./images/embds.png"
        vis_embds(embds, words_in_word2vec, train_unknown_words, 
                                dev_and_test_unknown_words, filename)
    trainer = Trainer()
    output = trainer.build(max_len, vocab_lookup, embds, args.n_classes, 
                           args.n_layers, args.fs, args.n_filters)
    trainer.train(output, padded_data, labels, args.n_epochs, args.bs, args.lr)
    if args.vis_embds != 0:
        filename = "./images/embds_new.png"
        vis_embds(embds, words_in_word2vec, train_unknown_words, 
                                dev_and_test_unknown_words, filename)


