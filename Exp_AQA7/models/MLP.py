import torch.nn as nn


class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block, self).__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        score = self.regressor(x)
        return score

if __name__ == '__main__':
    mlp = MLP_block(1024, 1)
    print(mlp.regressor)