import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        def init_rnn(self, rnn_type = 'LSTM'):
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size, hidden_size,
                                   bias = True,
                                   num_layers=2
                                   , batch_first=True)

            elif rnn_type == 'GRU':
                self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)


if __name__ == '__main__':
    model = nn.LSTM(128, 256, num_layers=2, batch_first=True)
    print(model)

                
        
        