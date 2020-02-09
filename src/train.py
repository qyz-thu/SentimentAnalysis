import torch
import torch.nn as nn
import numpy as np
import re
import time
import torch.optim as optim
import torch.nn.functional as F
import sys
MAX_LENGTH = 120     # max length of sentence
vocab_size = 0
data_path = "./train.txt"
emb_path = "./emb.npy"
vocab_path = "./vocab.txt"
lstm_path = "./lstm_net.pkl"
cnn_path = "./cnn_net.pkl"
fc_path = "./fc_net.pkl"
dimension = 300     # dimension of word embedding
batch_size = 10      # batch size
hidden_size = 50    # length of output in lstm layer
kernel_size1 = 3    # length of first filter
kernel_size2 = 4    # length of second filter
kernel_size3 = 5    # length of third filter
drop_out = 0.4      # drop out rate
iter_time = 50     # iteration times
learning_rate = 0.001    # learning rate
class_num = 8       # number of classes
optim_type = 1      # 0 for SGD, 1 for Adams
model_type = 1      # 1 for CNN, 2 for LSTM, else for baseline
use_regression = False   # use regression
trained_before = False   # trained before


def loadData():
    # prepare dictionary
    global vocab_size
    vocab = {'unk': 0}
    vocab_size += 1
    with open(vocab_path, 'r') as f:    # read in vocabulary
        words = f.read()
        words = re.findall("[\u4e00-\u9fa5]+", words)
        for word in words:
            vocab[word] = vocab_size
            vocab_size += 1

    # prepare embedding
    emb = np.load(emb_path)
    e = nn.Embedding(vocab_size, 300)
    emb = torch.from_numpy(emb)
    e.weight.data.copy_(emb)

    # prepare labels and sentences
    labels = []
    class_size = [0 for i in range(class_num)]
    class_sentences = [[] for i in range(class_num)]
    class_labels = [[] for i in range(class_num)]
    sentences = []
    with open(data_path, 'r') as f:
        i = 0
        for line in f:
            label = re.findall("[0-9]+", line)
            label = [int(n) for n in label]
            Max = 0
            M = label[0]
            for j in range(1, len(label)):
                if label[j] > M:
                    Max = j
                    M = label[j]
            class_size[Max] += 1
            label = np.array(label)
            class_labels[Max].append(label)
            labels.append(label)

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
            class_sentences[Max].append(sentence)
            i += 1

    for i in range(class_num):
        if i == 3:
            continue
        while class_size[i] < 2 * len(class_sentences[i]):
            index1 = np.random.randint(0, len(sentences))
            index2 = np.random.randint(0, len(class_sentences[i]))
            sentences.insert(index1, class_sentences[i][index2])
            labels.insert(index1, class_labels[i][index2])
            class_size[i] += 1

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

    return e, torch.tensor(labels).long(), sen


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
        output = self.classifier(input)
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


def train(emb, labels, sentences, batch_size=1, epochs=50, optim_type=0, model_type=1):
    if model_type == 2:
        net = LSTMNet(dimension, hidden_size)
        if trained_before:
            net.load_state_dict(torch.load(lstm_path))
    elif model_type == 1:
        net = CNNNet(dimension, hidden_size // 6)
        if trained_before:
            net.load_state_dict(torch.load(cnn_path))
    else:
        net = FCNet(dimension, hidden_size)
        if trained_before:
            net.load_state_dict(torch.load(fc_path))
    if optim_type == 0:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    CEloss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    start_time = time.time()
    for epoch in range(epochs):
        avg_loss = 0
        index = 0
        cur_time = time.time() - start_time
        h = cur_time // 3600
        cur_time %= 3600
        m = cur_time // 60
        cur_time %= 60
        print("epoch: %d, time used: %dh %dmin %ds" % (epoch, h, m, cur_time))
        count = 0
        while index < sentences.size()[0]:
            count += 1
            data_list = []
            label_list = []
            for i in range(batch_size):
                data_list.append(emb(sentences[index]))
                # id = 0
                if use_regression:
                    l = labels[index].cpu()
                    total = 0
                    for j in range(class_num):
                        total += l[j].numpy().tolist()
                    label_list.append(labels[index] / total)
                else:
                    id = 0
                    Max = labels[index][0]
                    for j in range(1, class_num):
                        if labels[index][j] > Max:
                            id = torch.tensor([j])
                            Max = labels[index][j]
                    label_list.append(id)

                index += 1
                if index >= sentences.size()[0]:
                    break
            data = torch.stack(tuple(data_list), dim=0).float().cuda()
            if use_regression:
                label = torch.stack(tuple(label_list), dim=0).float().cuda()
            else:
                label = torch.tensor(label_list).long().cuda()

            net.zero_grad()
            score = net(data)
            score = score.cuda()
            if use_regression:
                score = F.softmax(score, dim=1)
                current_loss = MSELoss(score, label).cuda()
            else:
                current_loss = CEloss(score, label).cuda()
            # if count % 10 == 0:
                # print(current_loss)
            avg_loss += current_loss.detach().cpu().numpy().tolist()
            current_loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            if model_type == 2:
                torch.save(net.state_dict(), lstm_path)
            elif model_type == 1:
                torch.save(net.state_dict(), cnn_path)
            else:
                torch.save(net.state_dict(), fc_path)
            print("model saved.")
        avg_loss /= index
        print("avg_loss = %.4f" % avg_loss)
        pass
    # save model parameters
    if model_type == 2:
        torch.save(net.state_dict(), lstm_path)
    elif model_type == 1:
        torch.save(net.state_dict(), cnn_path)
    else:
        torch.save(net.state_dict(), fc_path)
    print("model saved.")
    pass


emb, labels, sentences = loadData()
emb = emb.cuda()
labels = labels.cuda()
sentences = sentences.cuda()
if len(sys.argv) > 1:
    model_type = int(sys.argv[1])
train(emb, labels, sentences, batch_size=batch_size, epochs=iter_time,
      optim_type=optim_type, model_type=model_type)
