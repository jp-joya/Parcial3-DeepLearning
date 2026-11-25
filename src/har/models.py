import torch
from torch import nn


class HARModel(nn.Module):
    def __init__(self, architecture, hidden=64, dropout=0.25, layers=1,
                 conv_channels=32, kernel=5, **unused):
        super().__init__()
        if architecture not in ('rnn', 'lstm', 'cnn_lstm'):
            raise ValueError(architecture)
        self.architecture = architecture
        inputs = 9
        if architecture == 'cnn_lstm':
            self.conv = nn.Sequential(nn.Conv1d(9, conv_channels, kernel, padding=kernel // 2),
                                      nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout))
            inputs = conv_channels
        recurrent = nn.RNN if architecture == 'rnn' else nn.LSTM
        self.recurrent = recurrent(inputs, hidden, num_layers=layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
        # Dropout explícito: nn.LSTM ignora su dropout interno con una sola capa.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 6))

    def forward(self, x):
        if self.architecture == 'cnn_lstm':
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, state = self.recurrent(x)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.head(hidden[-1])
