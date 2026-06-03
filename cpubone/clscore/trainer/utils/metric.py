# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Conference on Computer Vision and Pattern Recognition (CVPR), 2025

import torch

__all__ = ["accuracy"]


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
