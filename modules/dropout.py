# -*- coding: utf-8 -*-

import torch



class sharedDropout(torch.nn.Module):

    def __init__(self, p=0.5, batchFirst=True):
        super(sharedDropout, self).__init__()

        self.p = p
        self.batchFirst = batchFirst

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batchFirst:
            s += f", batchFirst={self.batchFirst}"

        return s

    def forward(self, x):
        if self.training:
            if self.batchFirst:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batchFirst else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class independentDropout(torch.nn.Module):

    def __init__(self, p=0.5):
        super(independentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items