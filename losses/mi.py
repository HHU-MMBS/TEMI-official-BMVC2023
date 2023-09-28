from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

__all__ = [
    "DINOLossMI",
    "WMIBase",
    "PMI",
    "SCAN",
]

import utils
from losses.loss_utils import sim_weight, beta_mi


class StudentTeacherLoss(nn.Module, ABC):

    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.batchsize = batchsize
        self.out_dim = out_dim
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def student_probs(self, x):
        return F.softmax(x / self.student_temp, dim=-1)

    def teacher_probs(self, x, epoch=0):
        return F.softmax(x / self.teacher_temp_schedule[epoch], dim=-1)

    @abstractmethod
    def forward(self, student_out, teacher_out, epoch):
        pass

    def update_ema(self, output, ema, momentum, use_momentum=True):
        """
        Update exponential moving averages for teacher output.
        """
        if isinstance(output, tuple):
            output = torch.cat(output)
        batch_center = torch.sum(output, dim=0, keepdim=True)
        utils.all_reduce(batch_center)
        batch_center = batch_center / (len(output) * utils.get_world_size())
        if use_momentum:
            return ema * momentum + batch_center * (1 - momentum)
        else:
            return batch_center


class DINOLossMI(StudentTeacherLoss):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, probs_momentum=0.9):
        super().__init__(
            out_dim=out_dim,
            batchsize=batchsize,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            nepochs=nepochs,
            student_temp=student_temp
        )
        self.center_momentum = center_momentum
        self.probs_momentum = probs_momentum
        self.register_buffer("probs_pos", torch.ones(1, out_dim)/out_dim)
        self.register_buffer("probs_neg", torch.ones(1, out_dim)/out_dim)

    def forward(self, student_out, teacher_out, epoch):
        ncrops_student = len(student_out)//self.batchsize
        ncrops_teacher = len(teacher_out)//self.batchsize

        temp = self.teacher_temp_schedule[epoch]
        student_probs = F.softmax(student_out / self.student_temp, dim=-1).chunk(ncrops_student)
        teacher_probs = F.softmax(teacher_out / temp, dim=-1).detach().chunk(ncrops_teacher)

        with torch.no_grad():
            self.probs_pos = self.update_ema(teacher_probs, self.probs_pos,
                                             self.probs_momentum,
                                             use_momentum=True)

        p_k = self.probs_pos

        total_loss = 0.0
        count = 0
        for w in range(ncrops_teacher):
            for v in range(ncrops_student):
                if v == w:
                    # we skip cases where student and teacher operate on the same view for positive examples
                    continue

                loss = - torch.log(torch.sum(teacher_probs[w] * student_probs[v] / p_k, dim=-1))
                total_loss += loss.mean()
                count += 1

        total_loss /= count
        return total_loss


class WMIBase(DINOLossMI):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, reg=False, alpha=0.5, beta=1,
                 weight_mi=True,
                 positive_pmi=False):
        super().__init__(out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                         warmup_teacher_temp_epochs, nepochs, student_temp=student_temp,
                         center_momentum=center_momentum)
        self.out_dim = out_dim
        self.alpha = alpha
        self.reg = reg
        self.weight_mi = weight_mi
        self.min_mi = 0 if positive_pmi else -torch.inf
        self.beta = beta

    def loss(self, p1, p2, pk, beta):
        return beta_mi(p1, p2, pk, beta=beta, clip_min=self.min_mi)

    def weight(self, p1, p2):
        raise not NotImplementedError("weight function not implemented")

    def forward(self, student_out, teacher_out, epoch=0):
        ncrops_student = len(student_out)//self.batchsize
        ncrops_teacher = len(teacher_out)//self.batchsize

        temp = self.teacher_temp_schedule[epoch]
        beta = self.beta
        student_out = (student_out / self.student_temp).chunk(ncrops_student)
        teacher_probs = F.softmax(teacher_out / temp, dim=-1).detach().chunk(ncrops_teacher)

        with torch.no_grad():
            self.probs_pos = self.update_ema(teacher_probs, self.probs_pos,
                                             self.probs_momentum,
                                             use_momentum=True)
            p_k = self.probs_pos

        # weight between teacher pairs
        if self.weight_mi:
            pairs = 0
            weight = 0
            for v1 in range(ncrops_teacher):
                for v2 in list(range(ncrops_teacher))[v1 + 1:]:
                    weight += self.weight(teacher_probs[v1], teacher_probs[v2])
                    pairs += 1
            weight = weight/pairs
        else:
            weight = 1

        # Weighted MI objective
        total_loss = 0.0
        count = 0
        for w in range(ncrops_teacher):
            for v in range(ncrops_student):
                if v == w:
                    # we skip cases where student and teacher operate on the same view for positive examples
                    continue
                loss = weight*self.loss(teacher_probs[w], F.softmax(student_out[v], dim=-1), p_k, beta)
                total_loss += loss.mean()
                count += 1

        # self-entropy regularization
        if self.reg:
            total_aux_loss = 0.0
            for s in range(ncrops_student):
                aux_loss = torch.sum( F.softmax(student_out[s], dim=-1) *  F.log_softmax(student_out[s], dim=-1), dim=-1)
                total_aux_loss += aux_loss.mean() # batch-wise mean
            total_loss = total_loss / count + self.alpha * (total_aux_loss / ncrops_student)
        else:
            total_loss = total_loss / count
        return total_loss


class WMI(WMIBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weight(self, p1, p2):
        return sim_weight(p1, p2)


class PMI(WMIBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, weight_mi=False)


class SCAN(DINOLossMI):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, alpha=5):
        super().__init__(out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                         warmup_teacher_temp_epochs, nepochs, student_temp=student_temp,
                         center_momentum=center_momentum)
        self.out_dim = out_dim
        self.alpha = alpha

    def forward(self, student_out, teacher_out, epoch):
        ncrops_student = len(student_out)// self.batchsize
        ncrops_teacher = len(teacher_out)// self.batchsize

        student_probs = self.student_probs(student_out).chunk(ncrops_student)
        with torch.no_grad():
            teacher_probs = self.teacher_probs(teacher_out, epoch=epoch).chunk(ncrops_teacher)

        # Weighted MI objective
        total_loss = 0.0
        count = 0
        for w in range(ncrops_teacher):
            for v in range(ncrops_student):
                if v == w:
                    # we skip cases where student and teacher operate on the same view for positive examples
                    continue
                loss = -(teacher_probs[w] * student_probs[v]).sum(dim=-1).log()
                total_loss += loss.mean()
                count += 1

        # self-entropy regularization
        total_aux_loss = 0.0
        for pkx in student_probs:
            pk = pkx.mean(dim=0)
            total_aux_loss += (pk * pk.log()).sum()
        return total_loss / count + self.alpha * ( total_aux_loss / ncrops_student)
