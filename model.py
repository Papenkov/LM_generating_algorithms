import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    """Model for test. Model with weigths are located `./data/gru_lm.pth`"""
    def __init__(self, rnn, hid_dim: int, vocab_size: int, bidirectional: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.rnn = rnn
        self.linear = nn.Linear(hid_dim * 2 if bidirectional else hid_dim, hid_dim)
        self.head = nn.Linear(hid_dim, vocab_size)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(batch)
        output, _ = self.rnn(embeddings)

        linear = self.dropout(self.linear(self.activation(output)))
        prediction = self.head(self.activation(linear))

        return prediction