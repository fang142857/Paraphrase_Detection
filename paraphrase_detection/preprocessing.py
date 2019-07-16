import os
import re
import sys


remove_list = ",'.&" 

def tokenize(sentence):
    """
    Tokenize a sentence
    @sentence: Input sentence
    @return: List of tokens 
    """
    result = []
    #remove period, quotation mark and carriage return at the end of 
    #a sentence
    sentence = re.sub(r'\.\"?\r?$\n?', "", sentence).lower()
    #seperators: all punctuations except for those in remove_list
    tokens = re.split(r"[^\w" + remove_list + "]", sentence)	
    for token in tokens:
        #remove comma at the end of a word
        if token.endswith(","):	 
            token = re.sub(",$", "", token)
	#remove 's at the end of a word
        if token.endswith("'s"): 
            token = re.sub("'s$", "", token)
        #remove '' at the end of a word
        if token.endswith("''"):
            token = re.sub("''$", "", token)
        #remove ' at the end of a word
        if token.endswith("'"):
            token = re.sub("'$", "", token)
        #remove '' at the start of a word
        if token.startswith("''"):
            token = re.sub("^''", "", token)
        #remove ' at the start of a word
        if token.startswith("'"):
            token = re.sub("^'", "", token)
	#remove , at the end of a word
        if token.endswith(","):
            token = re.sub(",$", "", token)
        #exclude non-alphanumeric tokens
        if re.search("\w", token) is None:	
            token = ""
        if token:
            result.append(token)
    return result

def write_to_file(f, label, sentence1, sentence2):
    """
    @f: File stream where to write
    @label: Category label of a sentence pair
    @sentence1: Sentence 1 in the sentence pair
    @sentence2: Sentence 2 in the sentence pair
    """
    f.write(label + "\t")
    f.write(" ".join(sentence1))
    f.write("\t")
    f.write(" ".join(sentence2))
    f.write("\n")


if __name__=="__main__":
    if len(sys.argv) < 2:
        print("file path is missing")
    else:
        with open(sys.argv[1], 'r') as f1, open(
                "./corpus/tokenized_" + os.path.basename(
                    sys.argv[1]), 'w') as f2:
            data = f1.readlines()
            #tokenize for test files
            if "test" in sys.argv[1]:
                # keep the ratio in dev+test
                dev_pos = int(0.665*725)
                dev_neg = 725 - dev_pos
                with open("./corpus/tokenized_" + os.path.basename(
                    sys.argv[1]).replace("test", "dev"), 'w') as f3:
                    count_pos = 0
                    count_neg = 0
                    for line in data[1:]:
                        content = line.split("\t")
                        label = content[0]
                        sentence1 = tokenize(content[3])
                        sentence2 = tokenize(content[4])
                        if label == "1":
                            if count_pos < dev_pos:
                                write_to_file(f3, label, sentence1, 
                                              sentence2)
                                count_pos += 1
                            else:
                                write_to_file(f2, label, sentence1, 
                                              sentence2)
                        else:
                            if count_neg < dev_neg:
                                write_to_file(f3, label, sentence1, 
                                              sentence2)
                                count_neg += 1
                            else:
                                write_to_file(f2, label, sentence1, 
                                              sentence2)
            #tokenize for training files
            else:
                for line in data[1:]:
                    content = line.split("\t")
                    label = content[0]
                    sentence1 = tokenize(content[3])
                    sentence2 = tokenize(content[4])
                    write_to_file(f2, label, sentence1, sentence2)
            
						


