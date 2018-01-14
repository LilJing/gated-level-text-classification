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
            nn.Conv1d(200, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 56, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(224, self.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc(x)
        # output layer
        x = self.log_softmax(x)
        return x