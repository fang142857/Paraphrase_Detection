import codecs


def encode_label(label):
    """
    Encode category label to one-hot vector
    @label: Category label, 0 or 1
    @return: One-hot vector
    """
    return [1, 0] if int(label)==0 else [0,1]

def load_data(filepath):
    """
    Read data and labels from a given file
    @filepath: Path to the corpus file
    @return: Vocabulary list, sentence pairs, labels, maximal and minimal
             length of sentences
    """
    vocab = []
    labels = []
    max_len = 0
    min_len = 100
    sentences = [[], []]
    with codecs.open(filepath, "r", encoding="utf8") as f:
        for line in f.readlines():
            sample = line.split("\t")
            # labels
            labels.append(encode_label(sample[0]))
            # sentences
            s1 = sample[1].split(" ")
            s2 = sample[2].split(" ")
            sentences[0].append([w.strip() for w in s1])
            sentences[1].append([w.strip() for w in s2])
            # vocab
            for w in s1: vocab.append(w.strip())
            for w in s2: vocab.append(w.strip())
            # max length
            l = max(len(s1), len(s2))
            l2 = min(len(s1), len(s2))
            if l > max_len: max_len = l 
            if l2 < min_len: min_len = l2 
    print("vocab in dataset: {}".format(len(set(vocab))))
    print("max length in dataset: {}".format(max_len))
    print("min length in dataset: {}\n".format(min_len))
    return list(set(vocab)), sentences, labels, max_len, min_len                                   
                                                       
