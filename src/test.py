import torch
import numpy as np
import torch.nn as nn
import re
import torch.nn.functional as F
import math
from scipy.stats import pearsonr
import sys
MAX_LENGTH = 120     # max length of sentence
vocab_size = 0
data_path = "./test.txt"
emb_path = "./emb.npy"
vocab_path = "./vocab.txt"
lstm_path = "./lstm_net.pkl"
cnn_path = "./cnn_net.pkl"
fc_path = "./fc_net.pkl"
result_path = "./test_result.txt"
dimension = 300     # dimension of word embedding
hidden_size = 50    # length of output in lstm layer
neuron_number = 64  # number of neurons in fully connected layer
kernel_size1 = 3    # length of first filter
kernel_size2 = 4    # length of second filter
kernel_size3 = 5    # length of third filter
drop_out = 0.2      # drop out rate
class_num = 8       # number of classes
model_type = 0      # 0 for baseline, 1 for cnn, 2 for lstm
dev_set = False      # using validation set


def loadData():
    global vocab_size
    vocab = {'unk': 0}
    vocab_size += 1
    with open(vocab_path, 'r') as f:  # read in vocabulary
        words = f.read()
        words = re.findall("[\u4e00-\u9fa5]+", words)
        for word in words:
            vocab[word] = vocab_size
            vocab_size += 1

    emb = np.load(emb_path)
    e = nn.Embedding(vocab_size, 300)
    emb = torch.from_numpy(emb)
    e.weight.data.copy_(emb)
    labels = np.empty((3000, 8))
    sentences = []
    with open(data_path, 'r') as f:
        i = 0
        for line in f:
            label = re.findall("[0-9]+", line)
            label = [int(n) for n in label]
            label = np.array(label)
            labels[i] = label
            words = re.findall("[\u4e00-\u9fa5]+", line)
            sentence = []
            length = 0
            for w in words:
                if w in vocab:
                    sentence.append(w)
                else:
                    sentence.append('unk')
                length += 1
                if length > MAX_LENGTH:
                    break
            while len(sentence) < MAX_LENGTH:
                sentence.append('unk')
            sentences.append(sentence)
            i += 1
    w = np.empty(MAX_LENGTH)
    w = w.astype(np.int64)
    sen = np.empty((len(sentences), MAX_LENGTH))
    sen = sen.astype(np.int64)
    for i in range(len(sentences)):
        for j in range(MAX_LENGTH):
            w[j] = vocab[sentences[i][j]]
        sen[i] = w
        pass
    sen = torch.from_numpy(sen)
    sen = sen.long()
    labels = torch.from_numpy(labels).long()
    if dev_set:
        i = 0
        l = []
        s = []
        while i < sen.size()[0]:
            l.append(labels[i])
            s.append(sen[i])
            i += 3
        labels = torch.stack(tuple(l), dim=0).long()
        sen = torch.stack(tuple(l), dim=0).long()

    return e, labels, sen


class LSTMNet(nn.Module):
    def __init__(self, dimension, hidden_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=dimension, hidden_size=hidden_size, batch_first=True)
        self.lstm = nn.DataParallel(self.lstm).cuda()
        self.classifier = nn.Linear(hidden_size, class_num)
        self.classifier = nn.DataParallel(self.classifier).cuda()

    def forward(self, x):
        output, hidden = self.lstm(x)
        output = output.cuda()
        output = output.transpose(0, 1)[-1]
        out = F.dropout(output, p=drop_out).cuda()
        score = self.classifier(out)
        return score


class CNNNet(nn.Module):
    def __init__(self, dimension, hidden_size):
        super(CNNNet, self).__init__()
        self.filter11 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size1)
        self.filter11 = nn.DataParallel(self.filter11).cuda()
        self.filter12 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size1)
        self.filter12 = nn.DataParallel(self.filter12).cuda()
        self.filter21 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size2)
        self.filter21 = nn.DataParallel(self.filter21).cuda()
        self.filter22 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size2)
        self.filter22 = nn.DataParallel(self.filter22).cuda()
        self.filter31 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size3)
        self.filter31 = nn.DataParallel(self.filter31).cuda()
        self.filter32 = nn.Conv1d(in_channels=dimension, out_channels=hidden_size, kernel_size=kernel_size3)
        self.filter32 = nn.DataParallel(self.filter32).cuda()
        self.classifier = nn.Linear(hidden_size * 6, class_num)
        self.classifier = nn.DataParallel(self.classifier).cuda()

    def forward(self, x):
        x.transpose_(1, 2)
        h11 = self.filter11(x)
        h12 = self.filter12(x)
        h21 = self.filter21(x)
        h22 = self.filter22(x)
        h31 = self.filter31(x)
        h32 = self.filter32(x)
        pool1 = nn.MaxPool1d(h11.size()[2])
        pool2 = nn.MaxPool1d(h21.size()[2])
        pool3 = nn.MaxPool1d(h31.size()[2])
        h11 = pool1(h11).squeeze(2)
        h12 = pool1(h12).squeeze(2)
        h21 = pool2(h21).squeeze(2)
        h22 = pool2(h22).squeeze(2)
        h31 = pool3(h31).squeeze(2)
        h32 = pool3(h32).squeeze(2)
        input = F.relu(torch.cat((h11, h12, h21, h22, h31, h32), dim=1))
        input = F.dropout(input, p=drop_out)
        output = torch.sigmoid(self.classifier(input))

        return output


class FCNet(nn.Module):
    def __init__(self, dimension, hidden_size):
        super(FCNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())
        self.layer1 = nn.DataParallel(self.layer1).cuda()
        self.layer2 = nn.Linear(MAX_LENGTH, hidden_size)
        self.layer2 = nn.DataParallel(self.layer2).cuda()
        self.layer3 = nn.Linear(hidden_size, class_num)
        self.layer3 = nn.DataParallel(self.layer3).cuda()

    def forward(self, x):

        x = torch.sigmoid(self.layer1(x).squeeze())
        x = F.relu(self.layer2(x))
        x = F.dropout(x, p=drop_out)
        x = F.relu(self.layer3(x))
        return x


def test(emb, labels, sentences):

    if model_type == 1:
        net = CNNNet(dimension, hidden_size // 6)
        net.load_state_dict(torch.load(cnn_path))
    elif model_type == 2:
        net = LSTMNet(dimension, hidden_size)
        net.load_state_dict(torch.load(lstm_path))
    else:
        net = FCNet(dimension, hidden_size)
        net.load_state_dict(torch.load(fc_path))

    count = 0
    i = 0
    TP = [0 for i in range(class_num)]
    FP = [0 for i in range(class_num)]
    TN = [0 for i in range(class_num)]
    FN = [0 for i in range(class_num)]
    avg_corr = 0
    cn = 0
    for i in range(sentences.size()[0]):
        data = emb(sentences[i])
        id1 = 0
        Max = 0
        total = 0
        for j in range(class_num):
            total += labels[i][j].cpu().numpy().tolist()
            if labels[i][j] > Max:
                id1 = j
                Max = labels[i][j]
        print("target: %d" % id1, end=' ')
        target = torch.div(labels[i].cpu().float(), total)
        data.unsqueeze_(0)
        score = net(data)
        score.squeeze_()
        score = F.softmax(score, dim=0)
        correlation = pearsonr(score.detach().cpu().numpy(), target.numpy())

        id2 = 0
        Max = score[0]
        for j in range(1, class_num):
            if score[j] > Max:
                id2 = j
                Max = score[j]
        print("predicted: %d" % id2, end=' ')
        if not math.isnan(correlation[0]):
            avg_corr += correlation[0]
            cn += 1
            print("correlation: %.2f" % correlation[0])
        for j in range(class_num):
            if j == id2:
                if j == id1:
                    TP[j] += 1
                else:
                    FP[j] += 1
            else:
                if j == id1:
                    FN[j] += 1
                else:
                    TN[j] += 1
        if id1 == id2:
            count += 1

    with open(result_path, 'a+') as f_w:
        if model_type == 0:
            f_w.write("result for FC network:\n")
        elif model_type == 1:
            f_w.write("result for CNN network:\n")
        else:
            f_w.write("result for LSTM network:\n")

        f_score = 0
        for j in range(class_num):
            if TP[j] == 0:
                print("no accurate prediction for class %d" % j)
                f_w.write("no accurate prediction for class %d\n" % j)
                continue
            p = TP[j] / (TP[j] + FP[j])
            r = TP[j] / (TP[j] + FN[j])
            f = 2 * p * r / (p + r)
            f_score += f

        f_score /= class_num

        print("f_score: %.3f" % f_score)
        f_w.write("f_score: %.3f\n" % f_score)
        print("correct: %d/%d %.3f" % (count, i, count/i))
        f_w.write("correct: %d/%d %.3f\n" % (count, i, count/i))
        print("average correlation rate: %.2f" % (avg_corr / cn))
        f_w.write("average correlation rate: %.2f\n\n" % (avg_corr / i))
    pass


emb, labels, sentences = loadData()
emb = emb.cuda()
labels = labels.cuda()
sentences = sentences.cuda()
if len(sys.argv) > 1:
    model_type = int(sys.argv[1])
test(emb, labels, sentences)

