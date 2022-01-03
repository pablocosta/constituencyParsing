import torch

from modules.dropout import sharedDropout

class biLSTM(torch.nn.Module):
    
    def __init__(self, inputSize, hiddenSize, numLayers=1, dropout=0):
        super(biLSTM, self).__init__()
        
        self.inputSize  = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers  = numLayers
        self.dropout    = dropout

        self.fCells     = torch.nn.ModuleList()
        self.bCells     = torch.nn.ModuleList()
        
        #number of cells from forward and backwards lstm networks
        for _ in range(self.numLayers):
            self.fCells.append(torch.nn.LSTMCell(input_size=inputSize,
                                            hidden_size=hiddenSize))
            self.bCells.append(torch.nn.LSTMCell(input_size=inputSize,
                                            hidden_size=hiddenSize))
            inputSize = hiddenSize * 2

        self.resetParameters()
    
        
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.inputSize}, {self.hiddenSize}"
        if self.numLayers > 1:
            s += f", numLayers={self.numLayers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s
    
    def resetParameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                torch.nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                torch.nn.init.zeros_(param)
    def permuteHidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = torch.nn.modules.rnn.apply_permutation(hx[0], permutation)
        c = torch.nn.modules.rnn.apply_permutation(hx[1], permutation)

        return h, c
    
    def layerForward(self, x, hX, cell, batchSizes, reverse=False):
        hx0 = hxI = hX
        hxN, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = sharedDropout.get_mask(hx0[0], self.dropout)

        for t in steps:
            lastBatchSize, batchSize = len(hxI[0]), batchSizes[t]
            if lastBatchSize < batchSize:
                hxI = [torch.cat((h, ih[lastBatchSize:batchSize]))
                        for h, ih in zip(hxI, hx0)]
            else:
                hxN.append([h[batchSize:] for h in hxI])
                hxI = [h[:batchSize] for h in hxI]
            hxI = [h for h in cell(x[t], hxI)]
            output.append(hxI[0])
            if self.training:
                hxI[0] = hxI[0] * hid_mask[:batchSize]
        if reverse:
            hxN = hxI
            output.reverse()
        else:
            hxN.append(hxI)
            hxN = [torch.cat(h) for h in zip(*reversed(hxN))]
        output = torch.cat(output)

        return output, hxN
        
        
    def forward(self, sequence, hX=None):
        x, batchSizes = sequence.data, sequence.batch_sizes.tolist()
        batchSize = batchSizes[0]
        hN, cN = [], []

        if hX is None:
            ih = x.new_zeros(self.num_layers * 2, batchSize, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permuteHidden(hX, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batchSize, self.hidden_size)
        c = c.view(self.num_layers, 2, batchSize, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batchSize)
            if self.training:
                mask = sharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                    hx=(h[i, 0], c[i, 0]),
                                                    cell=self.f_cells[i],
                                                    batch_sizes=batchSize)
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                    hx=(h[i, 1], c[i, 1]),
                                                    cell=self.b_cells[i],
                                                    batch_sizes=batchSize,
                                                    reverse=True)
            x = torch.cat((x_f, x_b), -1)
            hN.append(torch.stack((h_f, h_b)))
            cN.append(torch.stack((c_f, c_b)))
        x = torch.nn.utils.rnn.PackedSequence(x,
                            sequence.batch_sizes,
                            sequence.sorted_indices,
                            sequence.unsorted_indices)
        hX = torch.cat(hN, 0), torch.cat(cN, 0)
        hX = self.permuteHidden(hX, sequence.unsorted_indices)

        return x, hX
