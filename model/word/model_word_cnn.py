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
            nn.Conv1d(200, 128, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc = nn.Linear(384, self.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        # x = x.transpose(1, 2).contiguous()
        x = x.contiguous()
        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc(x)
        # output layer
        x = self.log_softmax(x)
        return x