import torch

from modules import charLSTM, mlp, bert, biaffine, biLSTM
from modules.dropout import independentDropout, sharedDropout


class modelConstituency(torch.nn.Module):
    
    def __init__(self, args):
        super(modelConstituency, self).__init__()
        self.args = args
        # the embedding layer
        self.wordEmbed = torch.nn.Embedding(num_embeddings=args.nWords, embedding_dim=args.nEmbed)

        if args.feat == "char":
            self.featEmbed = charLSTM.charLSTM(nChars=args.nFeats, nEmbed=args.nCharEmbed, nOut=args.nFeatEmbed)

        elif args.feat == "bert":
            self.featEmbed = bert.bertEmbedding(model=args.bertModel, nLayers=args.nBertLayers, nOut=args.nFeatEmbed)
            
        else:
            self.featEmbed = torch.nn.Embedding(num_embeddings=args.nFeats, embedding_dim=args.nFeatEmbed)
        
        #droput to embeddings
        
        self.embedDropout = independentDropout(p=args.embedDropout)
        
        #build lstm layer
        
        self.lstm         = biLSTM.biLSTM(inputSize=args.nEmbed+args.nFeatEmbed, hiddenSize=args.nLstmHidden, numLayers=args.nLstmLayers, dropout=args.lstmDropout)
        self.lstmDropout  = sharedDropout(p=args.lstmDropout)
        
        # the MLP layers
        #spans layers
        self.mlpSpanL   = mlp.mlp(nIn=args.nLstmHidden*2, nOut=args.nMlpSpan, dropout=args.mlpDropout)
        self.mlpSpanR   = mlp.mlp(nIn=args.nLstmHidden*2, nOut=args.nMlpSpan, dropout=args.mlpDropout)
        #left label
        self.mlpLabelL  = mlp.mlp(nIn=args.nLstmHidden*2, nOut=args.nMlpLabel, dropout=args.mlpDropout)
        #right label
        self.mlpLabelR  = mlp.mlp(nIn=args.nLstmHidden*2, nOut=args.nMlpLabel, dropout=args.mlpDropout)
        
         # the Biaffine layers
        self.spanAttn  = biaffine.biaffineNet(nIn=args.nMlpSpan, biasX=True, biasY=False)
        self.labelAttn = biaffine.biaffineNet(nIn=args.nMlpLabel, nOut=args.nLabels, biasX=True, biasY=True)
        
        self.padIndex  = args.padIndex
        self.unkIndex  = args.unkIndex

    def forward(self, words, feats):
        
        batchSize, seqLen = words.shape
        # get the mask and lengths of given batch
        mask     = words.ne(self.padIndex)
        lens     = mask.sum(dim=1)
        extWords = words
        # set the indices larger than numEmbeddings to unkIndex
        if hasattr(self, 'pretrained'):
            extMask   = words.ge(self.wordEmbed.num_embeddings)
            extWords = words.masked_fill(extMask, self.unkIndex)
        
        # get outputs from embedding layers
        wordEmbed = self.wordEmbed(extWords)
        
        
        if hasattr(self, 'pretrained'):
            wordEmbed += self.pretrained(words)
            
        if self.args.feat == 'char':
            featEmbed = self.featEmbed(feats[mask])
            featEmbed = torch.nn.utils.rnn.pad_sequence(featEmbed.split(lens.tolist()), True)
            
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
        else:
            feat_embed = self.feat_embed(feats)
        
        wordEmbed, featEmbed = self.embedDropout(wordEmbed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((wordEmbed, featEmbed), dim=-1)

        x = torch.nn.utils.rnn.pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, True, total_length=seqLen)
        x = self.lstmDropout(x)

        xF, xB = x.chunk(2, dim=-1)
        x = torch.cat((xF[:, :-1], xB[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        spanL = self.mlpSpanL(x)
        spanR = self.mlpSpanR(x)
        labelL = self.mlpLabelL(x)
        labelR = self.mlpLabelR(x)

        # [batch_size, seq_len, seq_len]
        sSpan = self.spanAttn(spanL, spanR)
        # [batch_size, seq_len, seq_len, n_labels]
        sLabel = self.labelAttn(labelL, labelR).permute(0, 2, 3, 1)

        return sSpan, sLabel