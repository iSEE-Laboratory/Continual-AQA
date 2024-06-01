import torch.nn as nn


class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block, self).__init__()
        self.regressor = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(in_dim, 512),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        score = self.regressor(x)
        return score


    def add_one_head(self):
        in_features = self.regressor[4].in_features
        out_features = self.regressor[4].out_features
        weight = self.regressor[4].weight.data
        bias = self.regressor[4].bias.data
        new_out_features = 1 + out_features
        new_fc = nn.Linear(in_features, new_out_features)
        kaiming_normal_init(new_fc.weight)
        new_fc.weight.data[:out_features] = weight
        new_fc.bias.data[:out_features] = bias
        self.regressor[4] = new_fc
        print('Change output dim from {} to {}'.format(out_features, new_out_features))


if __name__ == '__main__':
    mlp = MLP_block(1024, 1)
    print(mlp.regressor)