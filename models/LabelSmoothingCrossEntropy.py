
import torch
import torch.nn as nn
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, classes=5):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_pred = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))