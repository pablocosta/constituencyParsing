import torch



class charLSTM(torch.nn.Module):

    def __init__(self, nChars, nEmbed, nOut, padIndex=0):
        super(charLSTM, self).__init__()

        self.nChars   = nChars
        self.nEmbed   = nEmbed
        self.nOut     = nOut
        self.padIndex = padIndex

        # the embedding layer
        self.embed = torch.nn.Embedding(num_embeddings=nChars,
                                  embedding_dim=nEmbed)
        # the lstm layer
        self.lstm = torch.nn.LSTM(input_size=nEmbed,
                            hidden_size=nOut//2,
                            batch_first=True,
                            bidirectional=True)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.nChars}, {self.nEmbed}, "
        s += f"n_out={self.nOut}, "
        s += f"pad_index={self.padIndex}"
        s += ')'

        return s

    def forward(self, x):
        mask = x.ne(self.padIndex)
        lens = mask.sum(dim=1)

        x = torch.nn.utils.rnn.pack_padded_sequence(self.embed(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)

        return hidden