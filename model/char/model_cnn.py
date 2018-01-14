import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.nclass = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(70, 56, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(18872, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(1024, 128)

        self.gru = nn.GRU(56, 28, num_layers=1,batch_first=True)

        self.fc = nn.Linear(128, self.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        # x = x.transpose(1, 2).contiguous()
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x