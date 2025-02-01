import torch.nn as nn

"""
Fusion layer
@Author Tianlin Yang
"""

class finalLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(finalLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim) # hidden layer to output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output