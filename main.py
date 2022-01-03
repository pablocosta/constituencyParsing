# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from model import modelConstituency
from cmd.cmd import cmd
from utils.corpus import Corpus
from utils.data import TextDataset, batchify
from utils.metric import Metric

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class trainModel(cmd):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='data/ptb/train.pid',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/ptb/dev.pid',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/ptb/test.pid',
                               help='path to test file')
        subparser.add_argument('--fembed', default=None,
                               help='path to pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')

        return subparser

    def __call__(self, args):
        super(trainModel, self).__call__(args)

        train = Corpus.load(args.ftrain, self.fields)
        dev = Corpus.load(args.fdev, self.fields)
        test = Corpus.load(args.ftest, self.fields)

        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(train.buckets)} buckets")

        print("Create the model")
        self.model = modelConstituency(args)
        print(f"{self.model}\n")
        self.model = self.model.to(args.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            self.train(train.loader)

            print(f"Epoch {epoch} / {args.epochs}:")
            loss, dev_metric = self.evaluate(dev.loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test.loader)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
        self.model = modelConstituency.load(args.model)
        loss, metric = self.evaluate(test.loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

implementar o parser de constituencia
https://github.com/yzhangcs/crfpar/tree/crf-constituency


https://github.com/yzhangcs/parser

https://github.com/yzhangcs/parser/blob/main/EXAMPLES.md