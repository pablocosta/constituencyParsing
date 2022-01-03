# -*- coding: utf-8 -*-

import torch
from transformers import BertModel
from modules.scalarMix import scalarMix


class bertEmbedding(torch.nn.Module):
    
    
    def __init__(self, model, nLayers, nOuts, requiresGrad=False):
        super(bertEmbedding, self).__init__()
        
        self.bert         = BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert         = self.bert.requires_grad_(requiresGrad)
        self.nLayers      = nLayers
        self.nOuts        = nOuts
        self.requiresGrad = requiresGrad
        self.hiddenSize   = self.bert.config.hidden_size
        
        self.scalarMix    = scalarMix(nLayers)
        if self.hiddenSize != nOuts:
            self.projection = torch.nn.Linear(self.hiddenSize, nOuts, False)
            

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.nLayers}, n_out={self.nOuts}"
        if self.requiresGrad:
            s += f", requires_grad={self.requiresGrad}"
        s += ')'

        return s
    
    def forward(self, subWords, bertLens, bertMask):
        #get bert mean representation
        #todo: sentece bert
        
        batchSize, seqLen = bertLens.shape
        #mask token
        mask = bertLens.get(0)
        if not(self.requiresGrad):
            self.bert.eval()
        _, _, bert = self.bert(subWords, attention_mask=bertMask)
        bert = bert[-self.nLayers:] 
        bert = self.scalarMix(bert)
        bert = bert[bertMask].split(bertLens[mask].tolist())
        bert = torch.stack([i.mean(0) for i in bert])
        bertEmbed = bert.new_zeros(batchSize, seqLen, self.hidden_size)
        bertEmbed = bertEmbed.masked_scatter_(mask.unsqueeze(-1), bert)
        if hasattr(self, 'projection'):
            bertEmbed = self.projection(bertEmbed)

        return bertEmbed
        
        