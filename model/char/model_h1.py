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


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.kernel_num = 256
        self.nclass = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(86016, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 56),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(56, 28)

        self.gru = nn.GRU(self.kernel_num, self.kernel_num, num_layers=1, batch_first=True)

        highway_layers = []
        for _ in range(args.highway_num):
            highway_layers.append(Highway(in_size=self.kernel_num, out_size=self.kernel_num))
        self.highway_layers = nn.Sequential(*highway_layers)

        self.fc = nn.Linear(28, self.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.highway_layers(x)
        x, _ = self.gru(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        x = self.fc(x)
        # output layer
        x = self.log_softmax(x)
        return x