from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def pad_sentences(sentences, max_len):
    """
    Pad sentences to the maximal length
    @sentences: 2d array of sentence pairs
    @max_len: Maximal length of a sentence in all datasets
    @return: 2d array of padded sentence pairs
    """
    for i in [0, 1]:
        for j in range(len(sentences[i])):
            s = sentences[i][j]
            diff = max_len - len(s)
            offset_left = int(diff / 2)
            if diff % 2 == 0:
                offset_right = offset_left           
            else:      
                offset_right = offset_left + 1
            s = [""]* offset_left + s + [""]*offset_right
            sentences[i][j] = s
    return sentences 

def transform(vocab_in_word2vec, train_unknown_words, 
              dev_and_test_unknown_words, embds):
    """
    Transform embeddings to a dataframe with labels
    @vocab_in_word2vec: All words that are representable by Google 
                        word2vec
    @train_unknown_words: Words in training set that are not 
                          representable by Google word2vec
    @dev_and_test_unknown_words: Words in development or test set that 
                                 are not representable by Google word2vec
    @embds: Embeddings of all words
    @return: Transformed embeddings with labels
    """
    init = [tf.global_variables_initializer(), tf.tables_initializer()]
    with tf.Session() as session:
        session.run(init)
        y = []
        for i in range(len(vocab_in_word2vec)):
            y.append(0)
        for i in range(len(train_unknown_words)):
            y.append(1)
        for i in range(len(dev_and_test_unknown_words)):
            y.append(2)
        for i in range(1):
            y.append(3)
        X = embds.eval()
        pca = PCA(n_components=2) 
        transformed = pd.DataFrame(pca.fit_transform(X))
        transformed['y'] = pd.Series(y, index=transformed.index)
        return transformed

def draw(transformed, filename):
    """
    Visualize 2d compressed representation of word embeddings
    @transformed: Transformed embeddings with labels
    """
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    ax = fig.add_subplot(111)    
    ax.scatter(transformed[transformed['y']==0][0], 
               transformed[transformed['y']==0][1], 
               label='Words in Word2Vec', c='yellow', alpha=1.0)
    ax.scatter(transformed[transformed['y']==1][0], 
               transformed[transformed['y']==1][1], 
               label='Unknown Words in Training Set', c='lightgreen', 
               alpha=1.0)
    ax.scatter(transformed[transformed['y']==2][0], 
               transformed[transformed['y']==2][1], 
               label='Unknown Words in Dev or Test Set', 
               c='blue', alpha=0.5)
    ax.scatter(transformed[transformed['y']==3][0], 
               transformed[transformed['y']==3][1], 
               label='Empty String for Padding', c='red', alpha=1.0)    
    plt.legend()
    plt.savefig(filename)

