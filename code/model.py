import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, n_hidden_layers=5,
                 activation=F.elu, batch_norm=None, dropout_prob=0.):
        super(DNN, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        self._fc = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        if batch_norm is not None:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.bn4 = nn.BatchNorm1d(hidden_dim)
            self.bn5 = nn.BatchNorm1d(hidden_dim)

            self._bn = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5]

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = self._fc[i](x)
            if self.batch_norm:
                x = self._bn[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return F.softmax(self.out(x), dim=1)


def init_he_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
