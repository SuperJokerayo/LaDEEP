import torch
import torch.nn as nn

class Motion_Parameters_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(1, 2)
        x = self.relu(self.fc2(x))
        return x
    

if __name__ == "__main__":
    x = torch.randn(30, 1, 6)
    model = Motion_Parameters_Extractor()
    y = model(x)

    print(y.shape)
