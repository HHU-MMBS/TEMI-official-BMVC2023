from abc import ABC, abstractmethod

import torch

from losses.mi import StudentTeacherLoss

__all__ = [
    'MultiHeadWMI',
    "is_multihead",
    "TEMI",
]

from losses import sim_weight, beta_mi


def multihead_loss(cls):
    """Marks a loss as multiclass compatible."""
    cls.IS_MULTIHEAD = True
    return cls


def is_multihead(cls):
    """Checks if a loss works with multiple heads"""
    return getattr(cls, "IS_MULTIHEAD", False)


@multihead_loss
class MultiHeadWeightProbBase(StudentTeacherLoss, ABC):

    def __init__(self, num_heads, probs_momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs_momentum = probs_momentum
        self.num_heads = num_heads
        self.register_buffer("pk", 1/self.out_dim * torch.ones(self.num_heads, self.out_dim))

    @abstractmethod
    def weight(self, pt1, pt2, epoch, idx):
        pass

    @abstractmethod
    def loss(self, ps, pt, epoch, idx):
        pass

    @property
    def pos_probs(self):
        return self.pk[0]

    def forward(self, student_out, teacher_out, epoch):
        # student_out and teacher_out should be lists of length num_heads
        multi_loss = []
        weight = self.compute_weight(epoch, teacher_out)

        for idx, (s, t) in enumerate(zip(student_out, teacher_out)):
            ncrops_student = len(s) // self.batchsize
            ncrops_teacher = len(t) // self.batchsize

            student_probs = self.student_probs(s).chunk(ncrops_student)
            with torch.no_grad():
                teacher_probs = self.teacher_probs(t, epoch).detach().chunk(ncrops_teacher)
            tmp_loss = 0
            count = 0
            for w in range(ncrops_teacher):
                for v in range(ncrops_student):
                    if v == w:
                        # we skip cases where student and teacher operate on the same view for positive examples
                        continue
                    tmp_loss += (weight * self.loss(student_probs[v], teacher_probs[w], epoch, idx)).mean()
                    count += 1
            multi_loss.append(tmp_loss / count)

        return torch.stack(multi_loss)

    def compute_weight(self, epoch, teacher_out):
        weight_total = 0
        weight_count = 0
        with torch.no_grad():
            for idx, t in enumerate(teacher_out):
                ncrops_teacher = len(t) // self.batchsize

                teacher_probs = self.teacher_probs(t, epoch).detach().chunk(ncrops_teacher)
                self.pk[idx] = self.update_ema(teacher_probs, self.pk[idx], self.probs_momentum, use_momentum=True)

                for v1 in range(ncrops_teacher):
                    for v2 in list(range(ncrops_teacher))[v1 + 1:]:
                        weight_total += self.weight(teacher_probs[v1], teacher_probs[v2], epoch, idx)
                        weight_count += 1
            weight = weight_total / weight_count
        return weight


class MultiHeadWMI(MultiHeadWeightProbBase):

    def __init__(self, beta, *args, positive_pmi=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.min_mi = 0 if positive_pmi else -torch.inf

    def weight(self, pt1, pt2, epoch, idx):
        raise NotImplementedError("Weighting is not implemented for this loss.")

    def loss(self, ps, pt, epoch, idx):
        pk = self.pk[idx]
        return beta_mi(ps, pt, pk, beta=self.beta, clip_min=self.min_mi)


class TEMI(MultiHeadWMI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weight(self, pt1, pt2, epoch, idx):
        return sim_weight(pt1, pt2)
