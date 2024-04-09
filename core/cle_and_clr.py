import torch
import torch.nn as nn

class Characteristic_Line_Extractor(nn.Module):
    def __init__(self, dim = 3, input_length = 300):
        super().__init__()
        self.dim = dim
        self.input_length = input_length
        self._make_module()

    def _make_module(self):
        self.conv1 = nn.Conv1d(self.dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(32 + 64, 64, 1)
        self.conv4 = nn.Conv1d(self.input_length, 64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        feature1 = x
        x = self.relu(self.conv2(x))
        feature2 = x

        x = torch.cat((feature1, feature2), axis = 1)

        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.conv4(x)
        return x
    

class Characteristic_Line_Reverser(nn.Module):
    def __init__(self, dim = 3, output_length = 300):
        super().__init__()
        self.dim = dim
        self.output_length = output_length
        self._make_module()

    def _make_module(self):
        self.conv1 = nn.Conv1d(64, self.output_length, 1)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.conv3 = nn.Conv1d(32, self.dim, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.output_length, self.output_length)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.fc1(x))
        return x
    
if __name__ == "__main__":
    x = torch.randn(30, 3, 300)
    m1 = Characteristic_Line_Extractor()
    m2 = Characteristic_Line_Reverser()
    print(x.shape)
    x = m1(x)
    print(x.shape)
    x = m2(x)
    print(x.shape)
