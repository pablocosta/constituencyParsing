import torch
import os
import path
from utils.embedding import embedding
from utils.algorithms import cky, crf
from utils.utils import bos, eos, pad, unk
from transformers import BertTokenizer
from utils.corpus import corpus, Treebank
from utils.field import (BertField, CharField, ChartField, Field,
                                RawField)
from utils.fn import build, factorize
from utils.metric import BracketMetric


class cmd(object):
    
    def __call__(self, args):
        self.args = args
        if not path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.trees = RawField('trees')
            self.words = Field('words', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            if args.feat == 'char':
                #char features
                self.feats = CharField('chars', pad=pad, unk=unk,
                                        bos=bos, eos=eos, fix_len=args.fixLen,
                                        tokenize=list)
            elif args.feat == 'bert':
                tokenizer  = BertTokenizer.from_pretrained(args.bertModel)
                self.feats = BertField('bert',
                                        pad='[PAD]',
                                        bos='[CLS]',
                                        eos='[SEP]',
                                        tokenize=tokenizer.encode)           
            else:
                self.feats  = Field('tags', bos=bos, eos=eos)
            self.charts = ChartField('charts')
            if args.feat in ('char', 'bert'):
                self.fields = Treebank(TREE=self.trees, WORD=(self.words, self.feats), CHART=self.charts)
            else:
                self.fields = Treebank(TREE=self.trees, WORD=self.words, POS=self.feats, CHART=self.charts)  
                
            train = corpus.load(args.fTrain, self.fields)
            if args.fEmbed:
                embed = embedding.load(args.fembed, args.unk)
            else:
                embed = None
                
            self.words.build(train, args.minFreq, embed)
            self.feats.build(train)
            self.charts.build(train)
            torch.save(self.fields, args.fields) 
        else:
            self.fields = torch.load(args.fields)
            self.trees = self.fields.TREE
            if args.feat in ('char', 'bert'):
                self.words, self.feats = self.fields.WORD
            else:
                self.words, self.feats = self.fields.WORD, self.fields.POS
            self.charts = self.fields.CHART
        self.criterion = torch.nn.CrossEntropyLoss()
        
        args.update({
            'nWords': self.words.vocab.n_init,
            'nFeats': len(self.feats.vocab),
            'nLabels': len(self.charts.vocab),
            'padIndex': self.words.pad_index,
            'unkIndex': self.words.unk_index,
            'bosIndex': self.words.bos_index,
            'eosIndex': self.words.eos_index
        })

        print(f"Override the default configs\n{args}")
        print(f"{self.trees}\n{self.words}\n{self.feats}\n{self.charts}")

        
    def train(self, loader):
        self.model.train()
        for trees, words, feats, (spans, labels) in loader:
            self.optimizer.zero_grad()
            
            batchSize, seqLen = words.shape
            lens = words.ne(self.args.padIndex).sum(1) - 1
            mask = lens.new_tensor(range(seqLen - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seqLen-1, seqLen-1).triu_(1)
            sSpan, sLabel = self.model(words, feats)
            loss, _ = self.getLoss(sSpan, sLabel, spans, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        totalLoss = 0
        metric    = BracketMetric()
        
        for trees, words, feats, (spans, labels) in loader:
            batchSize, seqLen = words.shape
            lens              = words.ne(self.args.padIndex).sum(1) - 1
            mask              = lens.new_tensor(range(seqLen - 1)) < lens.view(-1, 1, 1)
            mask              = mask & mask.new_ones(seqLen-1, seqLen-1).triu_(1)
            sSpan, sLabel     = self.model(words, feats)
            loss, sSpan       = self.get_loss(sSpan, sLabel, spans, labels, mask)
            preds             = self.decode(sSpan, sLabel, mask)
            preds             = [build(tree, [(i, j, self.charts.vocab.itos[label]) for i, j, label in pred]) for tree, pred in zip(trees, preds)]
            totalLoss         += loss.item()
            metric([factorize(tree, self.args.delete, self.args.equal) for tree in preds], [factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        totalLoss /= len(loader)

        return totalLoss, metric
    
    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        allTrees = []
        for trees, words, feats in loader:
            batchSize, seqLen = words.shape
            lens = words.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seqLen - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seqLen-1, seqLen-1).triu_(1)
            sSpan, sLabel = self.model(words, feats)
            if self.args.mbr:
                sSpan = crf(sSpan, mask, mbr=True)
            preds = self.decode(sSpan, sLabel, mask)
            preds = [build(tree, [(i, j, self.charts.vocab.itos[label]) for i, j, label in pred]) for tree, pred in zip(trees, preds)]
            allTrees.extend(preds)

        return allTrees
    
    def getLoss(self, sSpan, sLabel, spans, labels, mask):
        spanMask = spans & mask
        spanLoss, spanProbs = crf(sSpan, mask, spans, self.args.mbr)
        labelLoss = self.criterion(sLabel[spanMask], labels[spanMask])
        loss = spanLoss + labelLoss

        return loss, spanProbs

    def decode(self, sSpan, sLabel, mask):
        predSpans = cky(sSpan, mask)
        predLabels = sLabel.argmax(-1).tolist()
        preds = [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(predSpans, predLabels)]

        return preds