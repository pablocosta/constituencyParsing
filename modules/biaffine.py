import torch

class biaffineNet(torch.nn.Module):
    
    def __init__(self, nIn, nOut=1, biasX=True, biasY=True):
        super(biaffineNet, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.biasX = biasX
        self.biasY = biasY
        self.weight = torch.nn.parameter(torch.Tensor(nOut, nIn + biasX, nIn + biasY))
        self.resetParameters()
        
    def resetParameters(self):
        torch.nn.init.xavier_uniform_(self.weight)    
        
    def forward(self, x, y):
        if self.biasX:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.biasY:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batchSize, nOut, seqLen, seqLen]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)
        
        return s