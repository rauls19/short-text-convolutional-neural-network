import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_size, num_filter, num_classes, dropout) -> None:
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # check padding
        self.conv_1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filter, kernel_size=filter_size)
        self.pool_1d = nn.MaxPool1d(kernel_size=filter_size)
        self.dropout = nn.Dropout(dropout)
        
        in_lin = int(num_filter * ((embed_dim / filter_size) - 1))
        self.fc = nn.Linear(3000, num_classes) # based on the .view(size(0), -1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1d(x)
        x = torch.tanh(x)
        x = self.pool_1d(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
class CNN_STC(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_size, num_filter, num_classes, dropout) -> None:
        super(CNN_STC, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # check padding
        self.conv_1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filter, kernel_size=filter_size)
        self.pool_1d = nn.MaxPool1d(kernel_size=filter_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3000, num_classes) # based on the .view(size(0), -1)
    
    def folding(self, x):
        # Last array problem: Odd Tensors drop last array
        # Two approaches: - Concatenate last array
        #                 - Drop last array
        if x.size()[0] % 2 != 0:
            x = torch.stack([x[i] + x[i+1] for i in range(0, x.size()[0]-1, 2)]) # Every two rows
            x_last = x[x.size()[0]-1]
            x_last = x_last.view(1, x_last.size()[0], x_last.size()[1])
            x = torch.cat((x, x_last), 0)
        else:
            x = torch.stack([x[i] + x[i+1] for i in range(0, x.size()[0], 2)]) # Every two rows
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1d(x)
        x = torch.tanh(x)
        x = self.pool_1d(x)
        x = self.folding(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x