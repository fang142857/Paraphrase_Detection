downloaded from: https://www.microsoft.com/en-us/download/confirmation.aspx?id=52398


# data format:
one training/test instance per line consisting of
paraphrase_label \t sentence_1_id \t sentence_2_id \t sentence_1 \t setence_2
- where paraphrase_label is binary: 0: sentence_2 is NOT a paraphrase of sentence_1, 1_ sentence_2 is a paraphrase of sentence_1
- sentence_1_id, sentence_2_id are the ids of the two sentences, they can be ignored for this task
- sentence_1: text (untokenized)
- sentence_2: text (untokenized)

example:
1       1571093 1571028 Feelings about current business conditions improved substantially from the first quarter, jumping from 40 to 55.        Assessment of current business conditions improved substantially, the Conference Board said, jumping to 55 from 40 in the first quarter.

train: 4,076 sentence pairs (2,753 positive: 67.5%)
test: 1,725 sentence pairs (1,147 positive: 66.5%)

preprocessing:
- 725 senentences should be split from test set as development set, ratio of positive examples should be the same in dev+test (about 66.5%)
- tokenize sentence_1 and sentence_2
