import torch
import torch.nn as nn
import torch.nn.functional as F

#nn.Module
class word_CNN(nn.Module):
    def __init__(self, args):
        super(word_CNN, self).__init__()
        self.seq_len = args.seq_len
        self.nclass = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(200, 56, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(252, 56),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.gru = nn.GRU(56, 28, num_layers=1,batch_first=True)

        self.fc = nn.Linear(56, self.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        # x = x.transpose(1, 2).contiguous()
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        self.seq_len = len(x.transpose(0, 1))
        x = self.fc1(x)
        # linear layer
        x = self.fc(x)
        # output layer
        x = self.log_softmax(x)
        return x