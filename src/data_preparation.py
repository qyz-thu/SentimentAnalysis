import numpy as np
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

traindata_path = "../实验数据/sina/sinanews.train"
testdata_path = "../实验数据/sina/sinanews.test"
embdata_path = "../实验数据/word_emb/cn_emb.txt"
emb_path = "../实验数据/embedding.txt"
path = "../实验数据/Word60.model"
test_path = "../实验数据/test.txt"
train_path = "../实验数据/train.txt"
vocab_path = "../实验数据/vocab.txt"

paths = [[test_path, testdata_path], [train_path, traindata_path]]
vocab = {}
_vocab = {}

def prepareData():
    vocab_size = 0
    for j in range(2):
        with open(paths[j][0], 'w') as f_w:
            with open(paths[j][1], 'r', encoding='utf-8') as f:
                for l in f:
                    # time = re.search("[0-8]+", l)
                    labels = re.search("Total:[0-9]+([\u4e00-\u9fa5: 0-9]+)", l)
                    numbers = labels.group(1)
                    numbers = re.findall("[0-9]+", numbers)
                    label = [int(n) for n in numbers]

                    # label = [0 for i in range(8)]
                    # label[Max] = 1
                    for i in range(8):
                        f_w.write(str(label[i]) + '\t')
                    text = re.findall("([\u4e00-\u9fa5]+)", l)
                    for i in range(8, len(text)):
                        if text[i] not in vocab:
                            vocab[text[i]] = vocab_size
                            vocab_size += 1
                        f_w.write(text[i] + '\t')
                    f_w.write('\n')
                    pass


def prepareVocab():
    with open(vocab_path, 'w') as f_w:
        for word in _vocab:
            f_w.write(word + '\t')


def prepareEmbedding():
    emb = np.empty((60000, 300))
    emb[0] *= 0
    with open(embdata_path, 'r', encoding='utf-8') as f:
        count = 1
        with open(emb_path, 'w') as f_w:
            for text in f:
                word = re.search("[\u4e00-\u9fa5]+", text)
                if word is None:
                    continue
                if word.group() in vocab and word.group() not in _vocab:
                    _vocab[word.group()] = 1
                    numbers = re.findall("-*[0-9]+[0-9.]+", text)
                    number = [float(n) for n in numbers]
                    if len(number) > 300:
                        number = number[len(number) - 300:]
                    emb[count] = number
                    f_w.write(text)
                    count += 1
        print(count)
        emb = emb[:count]
        np.save("../实验数据/emb", emb)


prepareData()
prepareEmbedding()
prepareVocab()
