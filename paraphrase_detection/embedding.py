class Embeddings():

    def __init__(self, model, vocab): 
        """
        @embeddings: Pre-trained word2vec embeddings
        @vocab: The entire vocabulary
        @return: Words representable by word2vec,
                 Embeddings of the representable vocabulary,
                 Unknown words not representable by word2vec
        """
        self.model = model
        self.vocab = vocab
        self.words_in_word2vec = []
        self.embds_in_word2vec = []
        self.unknown_words = []

        for w in self.vocab:
            try:
                vec = self.model.word_vec(w)
                self.words_in_word2vec.append(w)
                self.embds_in_word2vec.append(vec)
            except KeyError:
                self.unknown_words.append(w)

    def get_words_in_word2vec(self):
        return self.words_in_word2vec

    def get_embds_in_word2vec(self):
        return self.embds_in_word2vec

    def get_unknown_words(self):
        return self.unknown_words


