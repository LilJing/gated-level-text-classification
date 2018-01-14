import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.trans_gate = nn.Linear(in_size, out_size)
        self.highway = nn.Linear(in_size, out_size)

    def forward(self, x):
        t = F.sigmoid(self.trans_gate(x))
        x = t * self.highway(x) + (1 - t) * x
        return x

class word_char_CNN(nn.Module):
    def __init__(self, args):
        super(word_char_CNN, self).__init__()
        self.kernel_num=56
        self.conv0 = nn.Conv1d(200, 200, kernel_size=6, stride=6)
        self.nclass = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(200, 56, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.gru = nn.LSTM(56, 28, num_layers=1,batch_first=True)

        highway_layers = []
        for _ in range(args.highway_num):
            highway_layers.append(Highway(in_size=self.kernel_num, out_size=self.kernel_num))
        self.highway_layers = nn.Sequential(*highway_layers)

        self.fc = nn.Linear(280, self.nclass)
        self.log_softmax = nn.LogSoftmax()

        self.embedding = nn.Embedding(
            num_embeddings=26,
            embedding_dim=200)

    def gated(self, x, y):
        g = F.sigmoid(self.fc1(y))
        return (1 - g) * y + g * x

    def forward(self, x, y):
        x=x.view(128,-1)
        x = self.embedding(x)
        x=x.transpose(1,2)
        x = self.conv0(x)
        x=x.transpose(1,2)
        y=y.transpose(1,2)
        x = self.gated(x, y)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.highway_layers(x)
        x, _ = self.gru(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # output layer
        x = self.log_softmax(x)
        return x