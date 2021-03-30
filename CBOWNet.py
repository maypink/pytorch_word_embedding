import torch

class CBOWNet(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWNet, self).__init__()

        self.block1 = torch.nn.Sequential (
            torch.nn.Embedding(vocab_size, embedding_dim),
            torch.nn.Linear(embedding_dim, vocab_size),
        )

    def forward(self, x):
        x = self.block1(x)
        x = torch.sum(x, dim=1)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

