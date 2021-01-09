from collections import OrderedDict, defaultdict, Counter

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import inspect


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm

class DebiasLossFn(nn.Module):
    """General API for our loss functions"""

    def forward(self, hidden, logits, bias, labels):
        """
        :param hidden: [batch, n_hidden] hidden features from the last layer in the model
        :param logits: [batch, n_answers_options] sigmoid logits for each answer option
        :param bias: [batch, n_answers_options]
          bias probabilities for each answer option between 0 and 1
        :param labels: [batch, n_answers_options]
          scores for each answer option, between 0 and 1
        :return: Scalar loss
        """
        raise NotImplementedError()

    def to_json(self):
        """Get a json representation of this loss function.

        We construct this by looking up the __init__ args
        """
        cls = self.__class__
        init = cls.__init__
        if init is object.__init__:
            return []  # No init args

        init_signature = inspect.getargspec(init)
        if init_signature.varargs is not None:
            raise NotImplementedError("varags not supported")
        if init_signature.keywords is not None:
            raise NotImplementedError("keywords not supported")
        args = [x for x in init_signature.args if x != "self"]
        out = OrderedDict()
        out["name"] = cls.__name__
        for key in args:
            out[key] = getattr(self, key)
        return out


class Plain(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        loss = F.binary_cross_entropy_with_logits(logits['answer'], labels)

        loss *= labels.size(1)
        return loss


class Focal(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # import pdb;pdb.set_trace()
        focal_logits=torch.log(F.softmax(logits['answer'],dim=1)+1e-5) * ((1-F.softmax(bias,dim=1))*(1-F.softmax(bias,dim=1)))
        loss=F.binary_cross_entropy_with_logits(focal_logits,labels)
        loss*=labels.size(1)
        return loss

class ReweightByInvBias(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # Manually compute the binary cross entropy since the old version of torch always aggregates
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits['answer'])
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
        weights = (1 - bias)
        loss *= weights  # Apply the weights
        return loss.sum() / weights.sum()


class BiasProduct(DebiasLossFn):
    def __init__(self, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(BiasProduct, self).__init__()
        self.constant_smooth = constant_smooth
        self.smooth_init = smooth_init
        self.smooth = smooth
        if smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        smooth = self.constant_smooth
        if self.smooth:
            smooth += F.sigmoid(self.smooth_param)

        # Convert the bias into log-space, with a factor for both the
        # binary outputs for each answer option
        bias_lp = torch.log(bias + smooth)
        bias_l_inv = torch.log1p(-bias + smooth)

        # Convert the the logits into log-space with the same format
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits['answer'])
        # import pdb;pdb.set_trace()

        # Add the bias
        log_prob += bias_lp
        log_one_minus_prob += bias_l_inv

        # Re-normalize the factors in logspace
        log_prob, log_one_minus_prob = renormalize_binary_logits(log_prob, log_one_minus_prob)

        # Compute the binary cross entropy
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)
        return loss


class LearnedMixin(DebiasLossFn):
    def __init__(self, w, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        # self.w=0
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.smooth = smooth
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        factor = self.bias_lin.forward(hidden)  # [batch, 1]
        factor = F.softplus(factor)

        bias = torch.stack([bias, 1 - bias], 2)  # [batch, n_answers, 2]

        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)

        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits['answer'])
        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in
        logits = bias + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
        bias_logprob = bias - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()
        return loss + self.w * entropy


class CCB_loss(DebiasLossFn):
    def __init__(self, w, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(CCB_loss, self).__init__()
        self.w = w
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.smooth = smooth

        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))

        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        factor_b = self.bias_lin.forward(hidden['fq'])  # [batch, 1]
        factor_b = F.softplus(factor_b)

        weights = (1 - bias)**1  #2-57.56   0.5-57.39

        bias = torch.stack([bias, 1 - bias], 2)  # [batch, the amount of answers, 2]
        # Smooth
        bias += self.constant_smooth

        if self.smooth:
            soften_factor_b = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor_b.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor_b.unsqueeze(1)

        log_prob_fq, log_one_minus_prob_fq = convert_sigmoid_logits_to_binary_logprobs(logits['fq'])
        log_probs_fq = torch.stack([log_prob_fq, log_one_minus_prob_fq], 2)

        log_prob_vq, log_one_minus_prob_vq = convert_sigmoid_logits_to_binary_logprobs(logits['vq'])
        log_probs_vq = torch.stack([log_prob_vq, log_one_minus_prob_vq], 2)


        # Add the bias in
        logits_ct = bias + log_probs_fq
        logits_cx = log_probs_vq
        logits_d = logits_ct + logits_cx


        # Renormalize to get log probabilities
        log_prob_ct, log_one_minus_prob_ct = renormalize_binary_logits(logits_ct[:, :, 0], logits_ct[:, :, 1])
        log_prob_cx, log_one_minus_prob_cx = renormalize_binary_logits(logits_cx[:, :, 0], logits_cx[:, :, 1])
        log_prob_d, log_one_minus_prob_d = renormalize_binary_logits(logits_d[:, :, 0], logits_d[:, :, 1])


        # Compute loss
        #loss_ct = -(log_prob_ct * labels + (1 - labels) * log_one_minus_prob_ct).sum(1).mean(0)
        loss_cx= -(log_prob_cx * logits['cx_label'] + (1 - logits['cx_label']) * log_one_minus_prob_cx).sum(1).mean(0)
        #logits_constant=torch.ones_like(logits['cx_label']).float()

        #loss_cx = -(log_prob_cx * logits_constant + (1 - logits_constant) * log_one_minus_prob_cx).sum(1).mean(0)

        loss_ct = -(log_prob_ct * labels + (1 - labels) * log_one_minus_prob_ct)
        loss_ct *= weights

        loss_d = -(log_prob_d * labels + (1 - labels) * log_one_minus_prob_d).sum(1).mean(0)

        loss=loss_ct.sum() / weights.sum()+loss_cx+loss_d
        #loss = loss_ct + loss_cx + loss_d  # (no reweight)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
        bias_logprob = bias - bias_norm.unsqueeze(2)
        # Compute and add the entropy penalty
        entropy_b = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()

        return loss + self.w * entropy_b