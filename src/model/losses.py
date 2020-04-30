# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7


class SimclrCriterion(nn.Module):
    '''
    Taken from: https://github.com/google-research/simclr/blob/master/objective.py

    Converted to pytorch, and decomposed for a clearer understanding.

    Args:
        init:
            batch_size (integer): Number of datasamples per batch.

            normalize (bool, optional): Whether to normalise the reprentations.
                (Default: True)

            temperature (float, optional): The temperature parameter of the
                NT_Xent loss. (Default: 1.0)
        forward:
            z_i (Tensor): Reprentation of view 'i'

            z_j (Tensor): Reprentation of view 'j'
    Returns:
        loss (Tensor): NT_Xent loss between z_i and z_j
    '''

    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SimclrCriterion, self).__init__()

        self.temperature = temperature
        self.normalize = normalize

        self.register_buffer('labels', torch.zeros(batch_size * 2).long())

        self.register_buffer('mask', torch.ones(
            (batch_size, batch_size), dtype=bool).fill_diagonal_(0))

    def forward(self, z_i, z_j):

        if self.normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)

        else:
            z_i_norm = z_i
            z_j_norm = z_j

        bsz = z_i_norm.size(0)

        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|

        Postives:
        Diagonals of ab and ba '\'

        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature

        # Compute Postive Logits
        logits_ab_pos = logits_ab[torch.logical_not(self.mask)]
        logits_ba_pos = logits_ba[torch.logical_not(self.mask)]

        # Compute Negative Logits
        logit_aa_neg = logits_aa[self.mask].reshape(bsz, -1)
        logit_bb_neg = logits_bb[self.mask].reshape(bsz, -1)
        logit_ab_neg = logits_ab[self.mask].reshape(bsz, -1)
        logit_ba_neg = logits_ba[self.mask].reshape(bsz, -1)

        # Postive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)

        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)

        neg = torch.cat((neg_a, neg_b), dim=0)

        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=1)

        loss = F.cross_entropy(logits, self.labels)

        return loss
